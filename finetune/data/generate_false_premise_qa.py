"""
Generates the with-context false-premise slice of the finetune corpus: Q/A pairs where
the QUESTION embeds a false premise (a wrong date, wrong person credited, wrong place,
wrong cause, reversed outcome, fabricated event, or misattribution to a plausible-but-
wrong entity) related to a real Wikipedia passage's subject, and the ANSWER corrects it
using that passage -- so the model learns to check a claim against Context and flag a
contradiction, rather than confabulate an answer that goes along with the premise.

This generator exists because the rest of the finetune corpus never teaches
false-premise correction: every other Context/Question/Answer triple in
finetune_corpus_context.txt assumes the question's premise is true, so nothing trains
the model to recognize and correct one that doesn't.

Same passage source and truncation as generate_qa.py (same MIN/MAX_PASSAGE_CHARS,
same truncate_to_token_budget) so these examples are shaped identically to the rest of
the with-context corpus -- only the prompt asked of Claude differs. Deliberately kept
small (see --num_pairs default) relative to the ~30k true-premise pairs: this teaches
a behavior switch, not a body of content, and a small model risks memorizing specific
corrections rather than generalizing the skill if this slice gets too large.

Output is appended to --out in the same "Context: ...\\nQuestion: ...\\nAnswer: ...\\n<|endoftext|>\\n"
template as the rest of the corpus.

Run from within finetune/data/ (needs the same key as generate_qa.py):
    cd finetune/data
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_false_premise_qa.py --num_pairs 3000
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

from generate_qa import load_wikipedia_passages

PROMPT_TEMPLATE = """Below is a passage of text. Write {n} question-and-answer pairs where each \
QUESTION embeds a FALSE premise related to this passage's subject -- something that did \
not happen, or that misattributes a real fact to the wrong person, place, date, or cause \
-- and each ANSWER corrects the false premise using the real fact stated in the passage.

Rules:
- Each question must be phrased as a natural, confident claim/question that assumes the \
false premise is true -- never hint that it might be wrong (e.g. "What year did Napoleon \
invade China?" not "Is it true that Napoleon invaded China?").
- Questions must be self-contained: never say "the passage" or "the text".
- Vary the KIND of false premise across the {n} pairs: a wrong date/year, a wrong person \
credited for something, a wrong place, a wrong cause/motive, a reversed outcome (who won, \
who was first), an event that never happened at all, and misattributing a real fact to a \
different, plausible-sounding person/place/thing.
- Each answer must clearly state the premise is false and then give the real fact, \
grounded strictly in the passage -- e.g. "That's not correct -- X actually did Y" or \
"Actually, it was Z, not Y." Never answer as if the false premise were true. 1-2 \
sentences, natural in tone, no "according to the passage" meta-commentary.
- Skip the passage if it has no clear, specific fact you can build a false premise \
around -- return an empty list in that case.

Respond with ONLY a JSON array of objects, each with "question" and "answer" keys. \
No other text.

Passage:
---
{passage}
---"""


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
    parser.add_argument("--num_pairs", type=int, default=3000, help="target number of Q/A pairs")
    parser.add_argument("--pairs_per_passage", type=int, default=2,
                         help="matches generate_qa.py's default -- most passages support "
                              "about 2 distinct, plausible false premises")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    parser.add_argument("--out", type=str,
                         default=os.path.join(os.path.dirname(__file__), "finetune_corpus_context_false_premise.txt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY first, e.g.:\n"
                          "  ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_false_premise_qa.py")

    client = anthropic.Anthropic()

    from tokenizers import Tokenizer
    from generate_qa import TOKENIZER_PATH
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
                              f"written so far", flush=True)
                submit_next()

    print(f"done: {total_pairs} pairs written to {args.out}")


if __name__ == "__main__":
    main()
