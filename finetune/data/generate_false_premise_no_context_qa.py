"""
Generates the no-context false-premise slice of the finetune corpus: Q/A pairs where
the QUESTION embeds a false premise about a well-known topic, and the ANSWER corrects
it -- but saved with "Context: N/A" (the same closed-book sentinel generate_no_context_qa.py
uses), so the model learns to catch an obviously-wrong premise even when retrieval found
nothing to hand it as Context.

The correcting fact still has to come from somewhere real, not from Claude's own
unaudited memory (see the project's Claude-role-boundary principle -- Claude writes the
Q/A shape, never invents the underlying fact). So this reuses the exact same grounding
approach as generate_false_premise_qa.py -- a real Wikipedia passage is shown to Claude
while it writes each pair -- but draws ONLY from rag/data/primary_articles/vital_passages.jsonl,
Wikipedia's own community-curated Vital Articles (Level 5) list, and explicitly asks for
premises about FAMOUS, widely-known aspects of the topic rather than obscure passage
details. The passage is then dropped from the saved training line: what's closed-book
at training time is restricted to genuinely well-known subjects, matching what a small
model could plausibly be expected to recognize as wrong without a retrieved passage.

Deliberately smaller than generate_false_premise_qa.py's context slice (see --num_pairs
default): closed-book false-premise correction is the harder, rarer case (most
false-premise questions in practice DO retrieve something), and this teaches a behavior
switch, not a body of content.

Output is appended to --out in the same "Context: N/A\\nQuestion: ...\\nAnswer: ...\\n<|endoftext|>\\n"
template generate_no_context_qa.py uses.

Run from within finetune/data/ (needs the same key as generate_qa.py):
    cd finetune/data
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_false_premise_no_context_qa.py --num_pairs 1000
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

DEFAULT_VITAL_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "rag", "data", "primary_articles", "vital_passages.jsonl")

PROMPT_TEMPLATE = """Below is a passage about {title}. Write {n} question-and-answer pairs where \
each QUESTION embeds a FALSE premise about this topic -- something that did not happen, or \
that misattributes a real fact to the wrong person, place, date, or cause -- and each ANSWER \
corrects the false premise using the real fact.

This will be used to train a model to catch the false premise from general knowledge alone, \
WITHOUT ever being shown this passage -- so only build premises around FAMOUS, widely-known \
facts about this topic (the kind of thing a general trivia or history question would test), \
never an obscure detail only this specific passage reveals.

Rules:
- Each question must be phrased as a natural, confident claim/question that assumes the \
false premise is true -- never hint that it might be wrong.
- Questions must be self-contained: never say "the passage" or "the text".
- Vary the KIND of false premise across the {n} pairs: a wrong date/year, a wrong person \
credited for something, a wrong place, a wrong cause/motive, a reversed outcome, an event \
that never happened at all, and misattributing a real fact to a different, plausible-sounding \
person/place/thing.
- Each answer must clearly state the premise is false and then give the real fact -- e.g. \
"That's not correct -- X actually did Y" or "Actually, it was Z, not Y." Never answer as if \
the false premise were true. 1-2 sentences, natural in tone.
- Answers must never mention "the passage" or "the text" -- state the real fact directly, \
as if from your own knowledge, since the model won't have any passage in front of it either.
- Skip if this topic has no famous, widely-known fact you can confidently build a false \
premise around without the passage -- return an empty list in that case.

Respond with ONLY a JSON array of objects, each with "question" and "answer" keys. \
No other text.

Passage:
---
{passage}
---"""


def load_vital_passages(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append((obj["title"], obj["passage"]))
    return entries


def generate_pairs_for_passage(client, model, title, passage, pairs_per_passage, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(n=pairs_per_passage, title=title, passage=passage)
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
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt * 5)
    text = next(b.text for b in response.content if b.type == "text").strip()
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
    parser.add_argument("--num_pairs", type=int, default=1000, help="target number of Q/A pairs")
    parser.add_argument("--pairs_per_passage", type=int, default=1,
                         help="1 per passage maximizes topic diversity across the corpus -- "
                              "with 45k vital passages available there's ample headroom for "
                              "skips/dedup to still reach --num_pairs")
    parser.add_argument("--vital_jsonl", type=str, default=DEFAULT_VITAL_JSONL)
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    parser.add_argument("--out", type=str,
                         default=os.path.join(os.path.dirname(__file__), "finetune_corpus_no_context_false_premise.txt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY first, e.g.:\n"
                          "  ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_false_premise_no_context_qa.py")

    client = anthropic.Anthropic()

    rng = random.Random(args.seed)
    entries = load_vital_passages(args.vital_jsonl)
    rng.shuffle(entries)
    print(f"loaded {len(entries)} vital passages from {args.vital_jsonl}")

    total_pairs = 0
    completed = 0
    seen_questions = set()
    write_lock = threading.Lock()
    stream_iter = enumerate(entries)

    def process(i, title, passage):
        try:
            pairs = generate_pairs_for_passage(client, args.model, title, passage, args.pairs_per_passage)
        except anthropic.APIError as e:
            print(f"  [passage {i}] API error, skipping: {e}")
            pairs = []
        return i, pairs

    with open(args.out, "a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending = set()

        def submit_next():
            if total_pairs >= args.num_pairs:
                return False
            try:
                i, (title, passage) = next(stream_iter)
            except StopIteration:
                return False
            pending.add(executor.submit(process, i, title, passage))
            return True

        for _ in range(args.concurrency):
            if not submit_next():
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                i, pairs = fut.result()
                with write_lock:
                    n_written = 0
                    for question, answer in pairs:
                        key = re.sub(r"\W+", "", question.lower())
                        if key in seen_questions:
                            continue
                        seen_questions.add(key)
                        out_f.write(f"Context: N/A\nQuestion: {question}\nAnswer: {answer}\n<|endoftext|>\n")
                        n_written += 1
                    out_f.flush()
                    total_pairs += n_written
                    completed += 1

                    if completed % 20 == 0 or total_pairs >= args.num_pairs:
                        print(f"  passage {i} | {completed} passages done | {total_pairs} pairs "
                              f"written so far", flush=True)
                submit_next()

    print(f"done: {total_pairs} pairs written to {args.out}")


if __name__ == "__main__":
    main()
