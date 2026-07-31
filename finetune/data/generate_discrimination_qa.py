"""
Generates a value-discrimination slice of the finetune corpus, targeted at the single
largest measured grounding failure in v6: the model copies a real value out of the
Context, but the wrong one of several of the same type.

The evidence (from test/runs/v6.jsonl's 97 finetuning_grounding_failure cases):

  - Of the failures whose answer contained a year, 89% used a year that IS present in
    the Context -- only 9% fabricated one. The contexts offered a median of 3 distinct
    years. So this is a *selection* failure, not a hallucination.
  - The wrong pick is often the subject's birth/death dates, the parenthetical Wikipedia
    opens biographies with -- "Who invented the light bulb?" -> "1847 ... 1931",
    "Who wrote Hamlet?" -> "lived from 1599 to 1601". The model reaches for the most
    salient numbers in the passage rather than the ones the question asked about.

Why the existing corpus doesn't teach this: its contexts are already distractor-rich
(median 3 distinct years, only 15% of year-answers have a unique year available), so
the gap isn't the passages -- it's the supervision. Of the contexts carrying two
year-bearing pairs in finetune_corpus_context.txt, 54% give the SAME year as the answer
to both questions. The model sees one passage, two questions, one identical date, over
and over: that trains "passage -> salient value", which is exactly the shortcut it uses
at inference. Nothing forces it to bind *this question* to *that value*.

This generator forces it. For each passage it extracts the distinct same-type values,
hands that list to Claude, and asks for one question per value whose unique correct
answer is that value -- so the same Context appears with several questions that must
resolve to *different* values. Every pair is then checked mechanically (see
validate_pair): the answer must contain its target value and must NOT contain any
sibling value, and the question must not contain the target value itself. Pairs failing
the check are dropped rather than trusted, and the rejection counts are reported -- the
prompt states the rule, the validator enforces it.

Answers are kept to one short sentence with no unsolicited extra detail. That's the
other confirmed v6 finding (test/smoketest_truncation.py): 24 of the 97 grounding
failures were *already correct* and failed only because of an appended wrong
embellishment -- a wrong nationality, a wrong movement count, wrong life-dates. Terse
answers and correct value-selection are complementary fixes, so this corpus applies
both rather than reintroducing the embellishment habit while fixing selection.

Passage pool: the same vital-articles passages the RAG index serves at inference
(rag/data/primary_articles/vital_passages.txt), so training and inference see identically
shaped text. Every passage that the test suite actually retrieved is excluded by default
(--exclude_contexts_from), so measured gains can't come from having memorized the
passages behind the benchmark's own questions.

Output is appended to --out in the same "Context: ...\\nQuestion: ...\\nAnswer: ...\\n<|endoftext|>\\n"
template as the rest of the corpus.

Run from within finetune/data/ (needs the same key as generate_qa.py):
    cd finetune/data
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_discrimination_qa.py \\
        --num_pairs 50 --out /dev/stdout        # sample first, eyeball the shape
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 generate_discrimination_qa.py \\
        --num_pairs 6000 --out finetune_corpus_context_discrimination.txt
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import anthropic

VITAL_PASSAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                               "rag", "data", "primary_articles", "vital_passages.txt")
# Every passage the retriever surfaces for the test question bank, from
# test/dump_test_passages.py -- 3,511 of them, against the 302 a single qa_loop.py run
# records as actually used. A run's own contexts are only its top-1 per question; a
# passage sitting at rank 2 answers the question just as well and would be just as
# contaminating, so the exclusion has to cover everything retrieval could surface.
DEFAULT_EXCLUDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "test", "runs", "test_suite_passages.jsonl")

# Value extractors, one per --value_type. Each returns the distinct values of that type
# in a passage, in order of appearance. Years are the primary axis because that's where
# the failure was measured, but the same failure shape shows up on quantities ("46 50
# square kilometers") and on same-category entities (Ferdinand III where the context
# said Ferdinand II, "Local Group" for Milky Way, "White Nile" for Nile) -- hence the
# extractor being pluggable rather than a hardcoded year regex.
EXTRACTORS = {
    "year": re.compile(r"\b(?:1[0-9]{3}|20[0-2][0-9])\b"),
    "number": re.compile(r"\b\d[\d,]*(?:\.\d+)?\b"),
}

# Answers must be terse but not degenerate. Asking only for brevity made Claude emit a
# bare "1880." for 64% of pairs in an early sample -- a shape that matches neither the
# rest of the corpus nor what chat.py shows a user, and that at corpus scale would teach
# the model to reply with a naked token. v6's *successful* answers ran ~12 words, so the
# target is a complete short sentence, and this is the floor the validator holds.
MIN_ANSWER_WORDS = 4

PROMPT_TEMPLATE = """Below is a passage. It contains these {value_type} values, all of \
which appear in it:

{values}

Write one question per value, in the same order, whose UNIQUE correct answer is that \
value. This trains a small model that currently grabs whichever value in a passage is \
most prominent instead of the one actually asked about, so these rules are the whole \
point of the exercise:

- Each question must be answerable from this passage with EXACTLY ONE of the listed \
values -- the one it is paired with. If a question could plausibly be answered by two \
of them, it is useless here; rewrite it to be specific enough that only one fits.
- The answer must state its value and must NOT mention any of the other listed values.
- Never put the answer's value in the question itself.
- Answers: ONE COMPLETE SHORT SENTENCE, typically 5-12 words, that restates enough of \
the question to stand on its own -- "Edison demonstrated the lamp in 1879.", never a bare \
"1879." A bare value is not an acceptable answer here.
- Within that sentence, state the asked-for fact and nothing else. Do not add a \
nationality, a date, a count, a translation, a parenthetical conversion, or any other \
detail the question did not ask for, even if the passage states it and even if it is \
true. Extra unrequested detail is a failure, not a bonus.
- Answer every question in the same style. Do not give a full sentence for one and a \
bare value for the next.
- Questions must be self-contained: never say "the passage" or "the text". Ask about the \
world directly.
- Vary the phrasing across the questions. Do not open every one with the same words -- \
mix "When did ...", "What year did ...", "In which year ...", "... took place in which \
year?", and embedded-clause forms. A model trained on one repeated question template \
learns the template rather than the skill.
- If this is a biography that opens with birth/death dates, at least one question must \
ask about something the person actually DID (when they invented/wrote/discovered/founded \
something), not when they were born or died -- confusing an achievement date with a life \
date is the specific error being corrected.
- If a listed value has no question that isolates it cleanly, skip it. Fewer good pairs \
beat padded ones -- return only the pairs you are confident in.

Respond with ONLY a JSON array of objects, each with "value", "question" and "answer" \
keys, where "value" is exactly the listed value that question targets. No other text.

Passage:
---
{passage}
---"""


def load_passages(path):
    """One passage per line -- vital_passages.txt is already cleaned, sentence-clipped
    and token-capped by rag/fetch_vital_articles.py via generate_qa.py's helpers, so
    these need no further processing to match the corpus's shape."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_excluded_contexts(path):
    """The exact passages the test suite retrieved, so they can be kept out of training.
    Generating from these would let a v7 score better by having memorized the benchmark's
    own supporting passages rather than by learning to select values."""
    if not path or not os.path.exists(path):
        return set()
    excluded = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            context = json.loads(line).get("context")
            if context:
                excluded.add(context.strip())
    return excluded


def distinct_values(passage, value_type, min_values):
    """The passage's distinct same-type values, or None if it hasn't got enough of them
    to pose a discrimination problem at all. Order of appearance is preserved so the
    prompt's list reads in the same order the passage does."""
    seen = []
    for match in EXTRACTORS[value_type].finditer(passage):
        value = match.group(0)
        if value not in seen:
            seen.append(value)
    return seen if len(seen) >= min_values else None


def validate_pair(pair, values, extractor):
    """Enforces what the prompt asked for. Returns a rejection reason, or None if the
    pair is good. The prompt states these rules; this is what actually holds them, since
    a pair that quietly violates them teaches precisely the confusion being corrected."""
    value, question, answer = pair["value"], pair["question"], pair["answer"]
    if value not in values:
        return "target value not one of the passage's extracted values"
    if value not in answer:
        return "answer does not contain its target value"
    siblings = {v for v in values if v != value}
    # Substring containment would fire on "1847" inside "18470"; match on extracted
    # value boundaries instead, the same way the values were pulled out to begin with.
    answer_values = set(extractor.findall(answer))
    intruders = answer_values & siblings
    if intruders:
        return f"answer also contains sibling value(s) {sorted(intruders)}"
    if value in question:
        return "question leaks its own answer"
    # Terseness is the goal, but a bare "1879." is a degenerate answer shape: it doesn't
    # match how the rest of the corpus (or chat.py's output) reads, and a few thousand of
    # them would teach the model to reply with a naked token. Floor as well as ceiling.
    if len(answer.split()) < MIN_ANSWER_WORDS:
        return f"answer is a bare value, under {MIN_ANSWER_WORDS} words"
    return None


def generate_pairs_for_passage(client, model, passage, values, value_type, extractor,
                                max_pairs=None, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(value_type=value_type, passage=passage,
                                     values="\n".join(f"- {v}" for v in values))
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.InternalServerError):
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt * 5)

    text = next(b.text for b in response.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return [], Counter({"unparseable JSON response": 1})

    kept, rejections = [], Counter()
    targeted = set()
    for item in raw:
        if not isinstance(item, dict) or not all(item.get(k) for k in ("value", "question", "answer")):
            rejections["malformed pair object"] += 1
            continue
        pair = {k: str(item[k]).strip() for k in ("value", "question", "answer")}
        reason = validate_pair(pair, values, extractor)
        if reason:
            rejections[reason] += 1
            continue
        # Two questions resolving to the same value would recreate the very pattern this
        # corpus exists to break (54% of the existing corpus's sibling pairs share an
        # answer year), so only the first question per value survives.
        if pair["value"] in targeted:
            rejections["duplicate target value within passage"] += 1
            continue
        targeted.add(pair["value"])
        kept.append((pair["question"], pair["answer"]))

    # A single pair from a passage teaches nothing contrastive -- the whole mechanism is
    # several questions over one Context resolving to different values.
    if len(kept) < 2:
        rejections["passage yielded <2 valid pairs, dropped"] += len(kept)
        return [], rejections
    # Value-dense passages can yield 8+ pairs each, which would spend the whole pair
    # budget on a few hundred Contexts. Contrast needs several questions per passage,
    # not all of them -- capping here buys far more distinct passages for the same
    # budget, and passage variety is what stops this becoming a template-learning set.
    if max_pairs is not None and len(kept) > max_pairs:
        rejections["over per-passage cap, trimmed"] += len(kept) - max_pairs
        kept = kept[:max_pairs]
    return kept, rejections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=6000, help="target number of Q/A pairs")
    parser.add_argument("--value_type", type=str, default="year", choices=sorted(EXTRACTORS),
                         help="which kind of same-type value the questions must discriminate "
                              "between -- 'year' is where the failure was measured")
    parser.add_argument("--max_pairs_per_passage", type=int, default=4,
                         help="cap on pairs kept per passage. Value-dense passages can "
                              "yield 8+, which would spend the budget on a few hundred "
                              "Contexts; capping buys more distinct passages, and passage "
                              "variety is what keeps this from becoming a template set")
    parser.add_argument("--min_values", type=int, default=3,
                         help="skip passages with fewer distinct values than this -- with "
                              "fewer there's no real selection problem to pose")
    parser.add_argument("--passages", type=str, default=VITAL_PASSAGES,
                         help="passage pool, one per line (default: the same vital-articles "
                              "passages the RAG index serves at inference)")
    parser.add_argument("--exclude_contexts_from", type=str, default=DEFAULT_EXCLUDE,
                         help="a qa_loop.py run whose retrieved contexts are excluded from "
                              "the passage pool, so training can't memorize the passages "
                              "behind the test suite's own questions. Pass '' to disable")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    parser.add_argument("--out", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "finetune_corpus_context_discrimination.txt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8,
                         help="passages in flight at once -- I/O-bound on the API, so "
                              "threads in one process are enough")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY first, e.g.:\n"
                          "  ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) "
                          "python3 generate_discrimination_qa.py")

    client = anthropic.Anthropic()
    extractor = EXTRACTORS[args.value_type]

    passages = load_passages(args.passages)
    print(f"loaded {len(passages)} passages from {args.passages}")

    excluded = load_excluded_contexts(args.exclude_contexts_from)
    if excluded:
        before = len(passages)
        passages = [p for p in passages if p.strip() not in excluded]
        print(f"excluded {before - len(passages)} passages retrieved by "
              f"{args.exclude_contexts_from}")

    eligible = [(p, v) for p in passages
                if (v := distinct_values(p, args.value_type, args.min_values))]
    print(f"{len(eligible)} passages have >={args.min_values} distinct {args.value_type} "
          f"values ({100 * len(eligible) / max(len(passages), 1):.0f}%)")
    if not eligible:
        raise SystemExit("no eligible passages -- try lowering --min_values")

    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    total_pairs = 0
    completed = 0
    rejections = Counter()
    write_lock = threading.Lock()
    stream_iter = enumerate(eligible)

    def process(i, passage, values):
        try:
            pairs, rej = generate_pairs_for_passage(client, args.model, passage, values,
                                                     args.value_type, extractor,
                                                     args.max_pairs_per_passage)
        except anthropic.APIError as e:
            print(f"  [passage {i}] API error, skipping: {e}")
            pairs, rej = [], Counter({"API error": 1})
        return i, passage, pairs, rej

    with open(args.out, "a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending = set()

        def submit_next():
            if total_pairs >= args.num_pairs:
                return False
            try:
                i, (passage, values) = next(stream_iter)
            except StopIteration:
                return False
            pending.add(executor.submit(process, i, passage, values))
            return True

        for _ in range(args.concurrency):
            if not submit_next():
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                i, passage, pairs, rej = fut.result()
                with write_lock:
                    for question, answer in pairs:
                        out_f.write(f"Context: {passage}\nQuestion: {question}\n"
                                    f"Answer: {answer}\n<|endoftext|>\n")
                    out_f.flush()
                    total_pairs += len(pairs)
                    completed += 1
                    rejections.update(rej)

                    if completed % 20 == 0 or total_pairs >= args.num_pairs:
                        print(f"  passage {i} | {completed} passages done | "
                              f"{total_pairs} pairs written | {sum(rejections.values())} rejected")
                submit_next()

    print(f"\ndone: {total_pairs} pairs from {completed} passages written to {args.out}")
    print(f"average {total_pairs / max(completed, 1):.1f} contrastive pairs per Context")
    if rejections:
        print("\nrejected by the validator (these never reach the corpus):")
        for reason, count in rejections.most_common():
            print(f"  {count:>5}  {reason}")


if __name__ == "__main__":
    main()
