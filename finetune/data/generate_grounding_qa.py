"""
Generates a grounding-reinforcement slice of the finetune corpus, targeted directly at
a demonstrated model failure mode rather than random new passages: context that's
retrieved and judged relevant, but whose answer doesn't correctly/specifically use it.
This failure mode breaks into four distinct sub-patterns, all addressed by this
generator's prompt:

  1. Fabricated specifics not stated in the passage (most common) -- e.g. inventing a
     "first Olympics since 1968" reading, or a fabricated name etymology, when the
     passage doesn't say that.
  2. Partial attribution loss -- e.g. crediting only one of two people the passage
     credits jointly (the Wright brothers -> "Orville Wright invented...").
  3. Failing the specific exclusion/comparative reasoning a question demands (e.g.
     "which of the following is NOT a dwarf planet: Pluto, Ceres, or Mercury?") even
     when the context supports picking the odd one out -- the model lists items
     instead of answering the actual question shape asked.
  4. Vague answers that miss the specific causal mechanism a "why/how" question needs,
     or that answer a different, more salient fact from the same passage instead of
     the one actually asked about.

Two passage sources, both real (see the project's Claude-role-boundary principle --
Claude only writes the Q/A shape, the fact always comes from a real passage):

  --failing_cases: known-failing (question, passage) pairs from a qa_loop.py run
    (default: test/runs/v2.jsonl), guaranteed included -- directly reinforces the
    demonstrated weak spots with several new, differently-phrased pairs per passage
    (never the original failing question verbatim, so this doesn't just teach the one
    memorized answer).
  general pool (default): the general wikipedia passage pool (load_wikipedia_passages,
    same as generate_qa.py), for volume and generalization -- the goal is a generalized
    "answer strictly and specifically from what's stated" skill, not a patch over a
    fixed set of specific questions.

Output is appended to --out in the same "Context: ...\\nQuestion: ...\\nAnswer: ...\\n<|endoftext|>\\n"
template as the rest of the corpus.

Run from within finetune/data/ (needs the same key as generate_qa.py):
    cd finetune/data
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_grounding_qa.py --num_pairs 5000
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

DEFAULT_FAILING_CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", "test", "runs", "v2.jsonl")

PROMPT_TEMPLATE = """Below is a passage of text. Write {n} factual question-and-answer pairs \
grounded STRICTLY in this passage -- someone who has only read your Q&A pairs, not the \
passage, should get correct answers, and every word of each answer must be traceable to \
something the passage actually states.

Rules -- these target real, observed failure modes of a small model, so follow them exactly:
- NEVER add a specific detail (a date, name, number, cause, translation, or claim) that \
isn't explicitly stated in the passage, even if it sounds plausible or you know it to be \
true from general knowledge. If the passage doesn't state it, leave it out of the answer.
- If the passage credits multiple people/things jointly with something, the answer must \
credit all of them, not just one.
- Vary question shape across the {n} pairs -- include at least one comparative/exclusion \
question if the passage supports it (e.g. "Which of the following was NOT ...: A, B, or \
C?", using one correct distractor from the passage and one plausible-but-wrong option NOT \
in the passage), and at least one "why/how" question if the passage states a specific \
cause or mechanism (not just an effect) -- the answer must give that specific mechanism, \
not a vague restatement of the effect.
- Questions must be self-contained: never say "the passage" or "the text".
- Answers should be 1-3 sentences, factual, no meta-commentary like "according to...".
- Skip the passage if it has no clear factual content, or no specific-enough content for \
a comparative/causal question -- return an empty list in that case, or fewer than {n} pairs.

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


def load_failing_passages(path):
    """Pulls the unique real passages behind v2's finetuning_grounding_failure
    cases -- deduped, since several failing questions can share the same passage."""
    seen = set()
    passages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("classification") != "finetuning_grounding_failure":
                continue
            passage = r.get("context")
            if passage and passage not in seen:
                seen.add(passage)
                passages.append(passage)
    return passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=5000, help="target number of Q/A pairs")
    parser.add_argument("--pairs_per_failing_passage", type=int, default=4,
                         help="pairs generated per known-failing passage -- several "
                              "differently-phrased pairs, never the original failing "
                              "question verbatim, so this teaches the skill rather than "
                              "memorizing one answer")
    parser.add_argument("--pairs_per_passage", type=int, default=2,
                         help="pairs per passage for the general wikipedia pool (volume/"
                              "generalization half) -- matches generate_qa.py's default")
    parser.add_argument("--failing_cases", type=str, default=DEFAULT_FAILING_CASES)
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    parser.add_argument("--out", type=str,
                         default=os.path.join(os.path.dirname(__file__), "finetune_corpus_context_grounding.txt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY first, e.g.:\n"
                          "  ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_grounding_qa.py")

    client = anthropic.Anthropic()
    rng = random.Random(args.seed)

    failing_passages = load_failing_passages(args.failing_cases)
    print(f"loaded {len(failing_passages)} unique known-failing passages from {args.failing_cases}")

    from tokenizers import Tokenizer
    from generate_qa import TOKENIZER_PATH
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    wikipedia_passages = load_wikipedia_passages(tokenizer)
    rng.shuffle(wikipedia_passages)
    print(f"loaded {len(wikipedia_passages)} general wikipedia passages")

    # Guarantee every failing passage is included (at its own higher pairs-per-passage),
    # then fill the rest of the target from the general pool.
    work_items = [(p, args.pairs_per_failing_passage) for p in failing_passages]
    work_items += [(p, args.pairs_per_passage) for p in wikipedia_passages]

    total_pairs = 0
    completed = 0
    write_lock = threading.Lock()
    stream_iter = enumerate(work_items)

    def process(i, passage, pairs_per_passage):
        try:
            pairs = generate_pairs_for_passage(client, args.model, passage, pairs_per_passage)
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
                i, (passage, ppp) = next(stream_iter)
            except StopIteration:
                return False
            pending.add(executor.submit(process, i, passage, ppp))
            return True

        for _ in range(args.concurrency):
            if not submit_next():
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                i, pairs = fut.result()
                passage = work_items[i][0]
                with write_lock:
                    for question, answer in pairs:
                        out_f.write(f"Context: {passage}\nQuestion: {question}\nAnswer: {answer}\n<|endoftext|>\n")
                    out_f.flush()
                    total_pairs += len(pairs)
                    completed += 1

                    if completed % 20 == 0 or total_pairs >= args.num_pairs:
                        print(f"  item {i} | {completed} passages done | {total_pairs} pairs "
                              f"written so far", flush=True)
                submit_next()

    print(f"done: {total_pairs} pairs written to {args.out}")


if __name__ == "__main__":
    main()
