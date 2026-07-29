"""
Generates a Q/A fine-tuning corpus by sampling Wikipedia passages from the local
raw-text sample (finetune/data/raw_sample/wikipedia) and asking Claude to write grounded
Q/A pairs for each one. Output is appended incrementally to --out, one triple per pair
in the "Context: ...\\nQuestion: ...\\nAnswer: ...\\n<|endoftext|>\\n" template the
fine-tune will train on, so tokenize_finetune.py can treat it exactly like the rest of the
corpus.

The passage is saved as "Context:" alongside its question/answer -- this is what makes
the finetune an *open-book* one: the model is trained to answer from a passage given to
it at inference time (via retrieval), rather than trying to recall facts from thin
parametric memory. See MAX_PASSAGE_TOKENS below for why passages are capped: the whole
Context+Question+Answer example has to fit inside the model's 1024-token block_size,
with room left over at inference time for the model to actually generate an answer.

Wikipedia-only: the finetune corpus (and retrieval, see rag_retrieve.py's module
docstring) excludes Gutenberg -- novels, first-person narratives, and bibliography
pages compete with, and often out-score, better Wikipedia matches purely on corpus
volume, making unreliable Context. Also generates a separate no-context, non-factual
half of the finetune corpus (see generate_no_context_qa.py) so the model learns the
"Context: N/A" sentinel rather than only ever seeing a real passage.

This module also owns the passage-cleaning functions the RAG side imports --
truncate_to_token_budget, clip_to_last_sentence, strip_pronunciation_guide (see each
one's own docstring). rag/embed_passages.py and rag/fetch_vital_articles.py call them
rather than reimplementing, so the passages the model reads at inference time are
shaped exactly like the ones it was finetuned on. Changing them changes both corpora.

Run from within finetune/data/ (see ../FINETUNE_PARAMS.md):
    cd finetune/data
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_qa.py --num_pairs 500
"""

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import anthropic
from tokenizers import Tokenizer

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw_sample")
WIKIPEDIA_DIR = os.path.join(RAW_DIR, "wikipedia")
# Same tokenizer as the base model -- only used here to precisely cap passage length
# in tokens (see MAX_PASSAGE_TOKENS below), not for training.
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "base", "data", "tokenizer", "tokenizer.json")

# Char-based bounds just decide roughly where to cut a passage out of the raw text --
# cheap, no tokenizing needed while scanning every article in ~2000 shards.
# Calibrated from this tokenizer's *measured* chars/token ratio on this corpus
# (2.5-4.8, averaging ~3.5 -- notably denser than the ~4 chars/token GPT-2 rule of
# thumb).
MIN_PASSAGE_CHARS = 500
MAX_PASSAGE_CHARS = 1200

# The real guarantee: after char-based extraction, every passage is precisely
# truncated (by actual token count, not the char estimate above) to this many tokens.
# 400 leaves ample room in the 1024-token block_size for "Context: "/"Question:
# "/"Answer: " labels, a typical question+answer (~50-70 tokens), and -- the part that
# actually matters at inference time -- generation headroom, since the model's sliding
# window (model.py's generate()) silently drops the *earliest* tokens once the total
# sequence exceeds block_size. If Context got dropped mid-generation because the
# prompt already used most of the budget, the model would lose its grounding partway
# through its own answer.
MAX_PASSAGE_TOKENS = 400

PROMPT_TEMPLATE = """Below is a passage of text. Write {n} factual question-and-answer pairs \
grounded strictly in this passage -- someone who has only read your Q&A pairs, not \
the passage, should get correct answers.

Rules:
- Questions must be self-contained: never say "the passage", "the text", or "the \
author" -- phrase questions as if asking about the world directly (e.g. "What year \
was Magna Carta sealed?" not "What year does the passage say Magna Carta was \
sealed?").
- Answers should be 1-3 sentences, factual, and match the register of the source \
material (no meta-commentary, no "According to...").
- Skip the passage if it has no clear factual content to ask about (e.g. pure table \
of contents, boilerplate) -- return an empty list in that case.
- Vary question types: who/what/when/where/why/how.

Respond with ONLY a JSON array of objects, each with "question" and "answer" keys. \
No other text.

Passage:
---
{passage}
---"""


_PRONUNCIATION_HINT = re.compile(
    r"pronounc|IPA"
    r"|[ɐ-ʯʰ-˿̀-ͯ]"  # IPA Extensions, Spacing Modifier
                                                    # Letters, Combining Diacritics
    r"|:\s*[^)]*[Ͱ-῿　-鿿가-퟿]"  # "Language: <non-Latin script>"
)
_LEADING_PAREN_RE = re.compile(r"^(\S+(?:\s\S+){0,3}?)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*")


def strip_pronunciation_guide(text):
    """Drops a leading parenthetical that's a pronunciation/foreign-script gloss (IPA
    transcription, "pronounced ...", or a "Language: <script>" aside) immediately after
    the subject -- e.g. "Stockholm (; Swedish: ˈstɔkː(h)ɔlm ) is ..." becomes
    "Stockholm is ...". Never touches a later or non-phonetic leading parenthetical
    (a birth-death date range, an abbreviation like "(NYC)") -- those carry real
    information the model should keep, unlike a phonetic gloss it was never trained on
    and can't use."""
    match = _LEADING_PAREN_RE.match(text)
    if match is None:
        return text
    if _PRONUNCIATION_HINT.search(match.group(2)):
        return match.group(1) + " " + text[match.end():]
    return text


def clip_to_last_sentence(text):
    """Drops a trailing partial sentence -- the char/token cutoffs above land wherever
    they land, with no awareness of word or sentence boundaries, so a passage can end
    mid-word (e.g. "...at which boil"). Cutting back to the last '.', '!', or '?'
    keeps every passage a clean, complete-sentence excerpt instead."""
    if text[-1:] in ".!?\"'":
        return text
    match = None
    for m in re.finditer(r'[.!?]["\']?\s', text):
        match = m
    if match is None:
        return text  # no earlier sentence boundary found -- nothing to clip back to
    return text[:match.end()].rstrip()


def truncate_to_token_budget(tokenizer, text):
    """Precisely caps a passage at MAX_PASSAGE_TOKENS -- the char-based bounds above
    only get the raw text roughly in range (the actual chars/token ratio varies
    2.5-4.8x across this corpus, so a char cutoff alone can't guarantee a token
    count). Also collapses internal whitespace/newlines to single spaces, so the
    saved "Context: {passage}" line stays one physical line -- simpler parsing, and
    matches how a retrieved passage would be handed to the model at inference time.
    Finally clips back to the last complete sentence (see clip_to_last_sentence) so
    neither the char nor the token cutoff can leave a passage ending mid-word."""
    text = re.sub(r"\s+", " ", text).strip()
    ids = tokenizer.encode(text).ids
    if len(ids) <= MAX_PASSAGE_TOKENS:
        return clip_to_last_sentence(text)
    return clip_to_last_sentence(tokenizer.decode(ids[:MAX_PASSAGE_TOKENS]))


def load_wikipedia_passages(tokenizer, wikipedia_dir=WIKIPEDIA_DIR):
    passages = []
    for fname in os.listdir(wikipedia_dir):
        path = os.path.join(wikipedia_dir, fname)
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for article in text.split("<|endoftext|>"):
            article = article.strip()
            if len(article) >= MIN_PASSAGE_CHARS:
                passages.append(truncate_to_token_budget(tokenizer, article[:MAX_PASSAGE_CHARS]))
    return passages


def generate_pairs_for_passage(client, model, passage, pairs_per_passage, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(n=pairs_per_passage, passage=passage)
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except anthropic.RateLimitError:
            # Concurrent workers make hitting the per-minute rate limit likely -- back
            # off and retry a few times rather than dropping the passage's pairs on the
            # first 429.
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt * 5)
    text = next(b.text for b in response.content if b.type == "text").strip()
    # Strip markdown code fences if the model wrapped the JSON in one.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        pairs = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [
        (p["question"].strip(), p["answer"].strip())
        for p in pairs
        if isinstance(p, dict) and p.get("question") and p.get("answer")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=500, help="target number of Q/A pairs")
    parser.add_argument("--pairs_per_passage", type=int, default=2,
                         help="distinct Q/A pairs to ask for per passage -- kept low since "
                              "passages are short (MAX_PASSAGE_TOKENS) and don't have enough "
                              "factual density to support many truly distinct questions")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "finetune_corpus.txt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8,
                         help="number of passages in flight at once -- this is I/O-bound "
                              "(waiting on the API), so threads within one process are "
                              "enough; no need for separate processes")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY first, e.g.:\n"
                          "  ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_qa.py")

    client = anthropic.Anthropic()
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    rng = random.Random(args.seed)
    wikipedia_passages = load_wikipedia_passages(tokenizer)
    rng.shuffle(wikipedia_passages)
    print(f"loaded {len(wikipedia_passages)} wikipedia passages")

    total_pairs = 0
    completed = 0
    write_lock = threading.Lock()
    stream_iter = enumerate(wikipedia_passages)

    def process(i, passage):
        try:
            pairs = generate_pairs_for_passage(client, args.model, passage, args.pairs_per_passage)
        except anthropic.APIError as e:
            print(f"  [passage {i}] API error, skipping: {e}")
            pairs = []
        return i, passage, pairs

    with open(args.out, "a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending = set()

        def submit_next():
            if total_pairs >= args.num_pairs:
                return False
            try:
                i, passage = next(stream_iter)
            except StopIteration:
                return False
            pending.add(executor.submit(process, i, passage))
            return True

        for _ in range(args.concurrency):
            if not submit_next():
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                i, passage, pairs = fut.result()
                with write_lock:
                    for question, answer in pairs:
                        out_f.write(f"Context: {passage}\nQuestion: {question}\nAnswer: {answer}\n<|endoftext|>\n")
                    out_f.flush()
                    total_pairs += len(pairs)
                    completed += 1

                    if completed % 20 == 0 or total_pairs >= args.num_pairs:
                        print(f"  passage {i} | {completed} passages done | {total_pairs} pairs "
                              f"written so far")
                submit_next()

    print(f"done: {total_pairs} pairs written to {args.out}")


if __name__ == "__main__":
    main()
