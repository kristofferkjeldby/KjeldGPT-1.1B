"""
Generates the non-factual, no-context half of the wikipedia-only finetune corpus: Q/A pairs
with no grounding passage at all, so the model learns to recognize "Context: N/A" (the
sentinel rag_retrieve.py/chat.py use when retrieval score falls below --min_context_score)
and respond conversationally/improvised instead of trying to extract an answer from
nonexistent context. Output is appended to --out in the same
"Context: ...\\nQuestion: ...\\nAnswer: ...\\n<|endoftext|>\\n" template generate_qa.py
uses, just with the Context line fixed to "N/A" -- one consistent template shape for
tokenize_finetune.py either way, rather than a differently-shaped prompt for the no-context
case (see chat.py's docstring on why: a fixed sentinel is a token-conditioning problem,
which a small model learns far more reliably than "notice a whole line is missing").

Deliberately NOT grounded on passages, and deliberately kept far smaller than the
wikipedia-context half of the corpus (10k): the goal here is to teach a behavior switch, not
a body of content, and a small model finetuned on a handful of epochs will start
memorizing specific Q->A mappings well before it needs to for that. Concretely, this
means:
  - every generated question should be meaningfully distinct (no near-duplicates
    across the whole corpus) -- forces the model to generalize the *mode switch*
    rather than recall a specific answer to a specific recurring question.
  - answers about identity/preferences/backstory deliberately invent DIFFERENT
    persona details across pairs rather than converging on one fixed persona -- the
    model currently improvises a different name every time it's asked "What is your
    name?" with no context, which is worth preserving rather than training away.

Run from within finetune/data/ (needs the same key as generate_qa.py):
    cd finetune/data
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_no_context_qa.py --num_pairs 2000
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import anthropic

# (weight, category description handed to the prompt) -- weights roughly mirror a
# realistic mix of what a chatbot actually gets asked when it's not a factual lookup.
CATEGORIES = [
    (20, "Personal identity questions -- name, age, appearance, background, family, "
         "pets, favorite things. Invent different, fresh persona details in each "
         "answer; never repeat the same name/age/backstory across pairs."),
    (20, "Questions about feelings, moods, and opinions on everyday topics -- weather, "
         "animals, art, food, fears, humor, optimism/pessimism. Answer with varied, "
         "genuine-sounding personal takes, not a consistent fixed personality."),
    (30, "Open-ended advice, hypothetical, or creative-prompt questions -- what would "
         "you do if..., tell me a joke/story, dream job, dinner ideas, superpowers, "
         "time travel. Answer playfully and inventively."),
    (20, "Casual conversation openers and small talk -- how was your day, what are "
         "you up to, how's the weather, plans for the weekend, did you sleep well."),
    (10, "Self-referential or existential questions about being an AI -- are you "
         "conscious, do you dream, are you real, do you get tired, do you get bored. "
         "Answer thoughtfully but briefly, without a fixed canonical stance repeated "
         "across pairs."),
]

PROMPT_TEMPLATE = """Write {n} diverse question-and-answer pairs for a small AI chatbot to train on.

These are NOT factual lookup questions -- there's no single correct answer, and the \
chatbot is expected to respond naturally and improvise rather than recall a specific \
fact. This batch's category:

{category}

Rules:
- Every question must be meaningfully different in wording and specificity from a \
cliche phrasing of the same idea -- vary it.
- Do not repeat a question, or write two that are near-duplicates of each other, \
within this batch.
- Answers should be 1-2 sentences, casual and natural in tone -- not preachy, not a \
list, not overly long.
- Where the category involves persona details (identity, preferences, backstory), \
invent something different and specific in each answer rather than reusing the same \
detail twice.

Respond with ONLY a JSON array of objects, each with "question" and "answer" keys. \
No other text."""


def generate_batch(client, model, category, batch_size, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(n=batch_size, category=category)
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=1.0,
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


def weighted_category_stream(categories):
    """Cycles through categories proportional to their weight, round-robin rather
    than in big blocks, so a rate-limit stall or early stop doesn't skew the mix."""
    counters = [0.0] * len(categories)
    while True:
        for i, (weight, desc) in enumerate(categories):
            counters[i] += weight
            if counters[i] >= 100:
                counters[i] -= 100
                yield desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=2000, help="target number of Q/A pairs")
    parser.add_argument("--batch_size", type=int, default=20,
                         help="pairs requested per API call -- no passage constraint here "
                              "(unlike generate_qa.py), so batching several pairs per call "
                              "is just more efficient")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    parser.add_argument("--out", type=str,
                         default=os.path.join(os.path.dirname(__file__), "finetune_corpus_no_context.txt"))
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY first, e.g.:\n"
                          "  ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_no_context_qa.py")

    client = anthropic.Anthropic()

    total_pairs = 0
    completed = 0
    seen_questions = set()
    write_lock = threading.Lock()
    category_iter = enumerate(weighted_category_stream(CATEGORIES))

    def process(i, category):
        try:
            pairs = generate_batch(client, args.model, category, args.batch_size)
        except anthropic.APIError as e:
            print(f"  [batch {i}] API error, skipping: {e}")
            pairs = []
        return i, pairs

    with open(args.out, "a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending = set()

        def submit_next():
            if total_pairs >= args.num_pairs:
                return False
            i, category = next(category_iter)
            pending.add(executor.submit(process, i, category))
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

                    if completed % 10 == 0 or total_pairs >= args.num_pairs:
                        print(f"  batch {i} | {completed} batches done | {total_pairs} pairs "
                              f"written so far ({len(pairs) - n_written} dupes dropped this batch)",
                              flush=True)
                submit_next()

    print(f"done: {total_pairs} pairs written to {args.out}")


if __name__ == "__main__":
    main()
