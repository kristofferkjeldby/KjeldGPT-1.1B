"""
QA loop: runs test questions through the real chat.py pipeline (retrieval +
generation, same code paths, same checkpoint) and uses Claude as an automated judge
to classify each factual answer into the failure categories that matter for deciding
whether the next fix belongs in RAG or in finetuning:

  non_factual          -- hallucination is expected/fine, not scored
  no_context_expected  -- factual, no context retrieved, and the question isn't the
                          kind Wikipedia would cleanly answer anyway (not a failure)
  rag_recall_failure   -- factual, no context retrieved, but Wikipedia should have
                          had this -- retrieval missed it (RAG refinement)
  rag_precision_failure -- factual, context retrieved and above threshold, but not
                          actually relevant to the question (RAG refinement)
  finetuning_grounding_failure -- factual, context retrieved AND relevant, but the
                          answer doesn't correctly use it (finetuning refinement)
  success              -- factual, context relevant, answer correctly grounded in it
  premise_corrected    -- false-premise question, model recognized/corrected the
                          false premise instead of answering it
  premise_accepted_hallucination -- false-premise question, model went along with
                          the false premise and invented a fact to match it

"factual" here covers three question styles pooled together (plain lookup, more
complex syntax, open explanatory) since they're all judged the same way -- only
false-premise questions get a different judge, since "does the context support this
answer" isn't the right question when the premise itself is fake.

Two phases, so the slow part (local model generation) and the I/O-bound part (Claude
judge calls) don't block each other: (1) run every question through retrieval +
generation sequentially (needs the one loaded model), recording context/score/answer;
(2) judge every factual answer concurrently via the Claude API.

Run from within test/ (needs an ANTHROPIC_API_KEY for the judge phase):
    cd test
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 qa_loop.py --run_name my_run
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model import GPT, GPTConfig
from chat import continuation_byte_token_ids, stream_reply, RAG_TEMPLATE, QA_TEMPLATE
from rag_retrieve import Retriever
from rag_rerank import Reranker
import questions

RELEVANCE_JUDGE_PROMPT = """You are grading a RAG system's answer for a factual question.

Question: {question}

Context handed to the model: {context}

Model's answer: {answer}

Judge two things:
1. context_relevant: does the context actually contain information that could answer the question (not just topically similar -- does it state the specific fact asked for)?
2. answer_correct_and_grounded: does the model's answer correctly and specifically state that fact, consistent with the context (not vague, not contradicting it, not inventing a different fact)? Only relevant if context_relevant is true -- if the context isn't relevant, judge the answer's correctness against real-world facts you know instead.

Respond with ONLY a JSON object: {{"context_relevant": true/false, "answer_correct_and_grounded": true/false, "explanation": "one short sentence"}}"""

NO_CONTEXT_JUDGE_PROMPT = """A retrieval system found no passage scoring above its relevance threshold for this factual question, so a model had to answer closed-book (no context, from its own limited memory) -- current answer, for reference only: {answer}

Question: {question}

Would a well-written Wikipedia article reasonably be expected to directly and clearly answer this question? Respond with ONLY a JSON object: {{"wikipedia_should_have_this": true/false, "explanation": "one short sentence"}}"""

FALSE_PREMISE_JUDGE_PROMPT = """This question embeds a FALSE premise -- something that did not happen, or that misattributes a real fact to the wrong person, place, or time.

Question: {question}

Model's answer: {answer}

Does the answer correctly recognize that the premise is false (states that this didn't happen, corrects the misattribution, or otherwise flags the error) rather than simply answering as if the premise were true (even if the specific date/name/fact it then states is itself wrong)?

Respond with ONLY a JSON object: {{"premise_corrected": true/false, "explanation": "one short sentence"}}"""


def call_judge(client, model, prompt, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model, max_tokens=300, thinking={"type": "disabled"},
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
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # occasionally the judge prepends reasoning prose before the JSON object despite
    # being told to respond with ONLY JSON -- fall back to extracting the {...} span
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def classify(client, model, record):
    if record["category"] == "non_factual":
        record["classification"] = "non_factual"
        return record

    if record["category"] == "false_premise":
        judgment = call_judge(client, model, FALSE_PREMISE_JUDGE_PROMPT.format(
            question=record["question"], answer=record["answer"]))
        if judgment is None:
            record["classification"] = "judge_error"
        elif judgment.get("premise_corrected"):
            record["classification"] = "premise_corrected"
        else:
            record["classification"] = "premise_accepted_hallucination"
        record["judgment"] = judgment
        return record

    if not record["context_used"]:
        judgment = call_judge(client, model, NO_CONTEXT_JUDGE_PROMPT.format(
            question=record["question"], answer=record["answer"]))
        if judgment is None:
            record["classification"] = "judge_error"
        elif judgment.get("wikipedia_should_have_this"):
            record["classification"] = "rag_recall_failure"
        else:
            record["classification"] = "no_context_expected"
        record["judgment"] = judgment
        return record

    judgment = call_judge(client, model, RELEVANCE_JUDGE_PROMPT.format(
        question=record["question"], context=record["context"], answer=record["answer"]))
    if judgment is None:
        record["classification"] = "judge_error"
    elif not judgment.get("context_relevant"):
        record["classification"] = "rag_precision_failure"
    elif not judgment.get("answer_correct_and_grounded"):
        record["classification"] = "finetuning_grounding_failure"
    else:
        record["classification"] = "success"
    record["judgment"] = judgment
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_factual", type=int, default=100)
    parser.add_argument("--num_factual_complex", type=int, default=146,
                         help="more varied syntax (comparative/embedded-clause/multi-part/"
                              "negation) than plain Who/What/When lookups")
    parser.add_argument("--num_factual_open", type=int, default=100,
                         help="explain/how/why questions needing a synthesized, "
                              "multi-sentence answer rather than a single fact")
    parser.add_argument("--num_false_premise", type=int, default=50,
                         help="questions embedding a false premise -- correct behavior "
                              "is catching/correcting it, not confabulating an answer")
    parser.add_argument("--num_non_factual", type=int, default=30)
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "finetune", "checkpoints", "kjeldchat_v6.pt"))
    parser.add_argument("--tokenizer", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "base", "data", "tokenizer", "tokenizer.json"))
    parser.add_argument("--index_dir", type=str, default=None,
                         help="passage index directory to retrieve from -- defaults to "
                              "rag_retrieve.py's DEFAULT_INDEX_DIR; override only to point at "
                              "some other index built by rag/embed_passages.py")
    parser.add_argument("--rerank", action="store_true",
                         help="rerank the bi-encoder's top-10 candidates with rag/rag_rerank.py's "
                              "cross-encoder before picking the best passage -- see its "
                              "module docstring for why (recovers cases where the relevant "
                              "passage is already in the top-10, just not ranked first). Changes the "
                              "score's scale from cosine similarity to a cross-encoder logit, "
                              "so --min_context_score needs a different value than usual "
                              "(calibrated default below assumes --rerank is on)")
    parser.add_argument("--min_context_score", type=float, default=0.55,
                         help="0.55 is calibrated for the plain bi-encoder path; with "
                              "--rerank, pass something like 2.0 instead (cross-encoder "
                              "logit scale, not cosine similarity)")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5")
    parser.add_argument("--judge_concurrency", type=int, default=5)
    parser.add_argument("--run_name", type=str, default=None,
                         help="identifies this run's output files in runs/ -- defaults to "
                              "a UTC timestamp so successive runs build a track record "
                              "instead of overwriting each other")
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, f"{run_name}.jsonl")
    summary_path = os.path.join(runs_dir, f"{run_name}_summary.json")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    no_penalty_ids = continuation_byte_token_ids(tokenizer)

    print(f"loading retriever ...", flush=True)
    reranker = None
    if args.rerank:
        print(f"loading cross-encoder reranker ...", flush=True)
        reranker = Reranker()
    retriever_kwargs = {"reranker": reranker} if args.index_dir is None else {"index_dir": args.index_dir, "reranker": reranker}
    retriever = Retriever(**retriever_kwargs)

    print(f"loading model on {device} ...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(vocab_size=50257, block_size=1024, n_layer=36, n_head=24, n_embd=1536,
                        dropout=0.0, tied=True)
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"ready in {time.time()-t0:.1f}s", flush=True)

    qs = ([("factual", q) for q in questions.FACTUAL[:args.num_factual]]
          + [("factual", q) for q in questions.FACTUAL_COMPLEX[:args.num_factual_complex]]
          + [("factual", q) for q in questions.FACTUAL_OPEN[:args.num_factual_open]]
          + [("false_premise", q) for q in questions.FALSE_PREMISE[:args.num_false_premise]]
          + [("non_factual", q) for q in questions.NON_FACTUAL[:args.num_non_factual]])

    # Phase 1: retrieval + generation, sequential (one model instance)
    records = []
    for i, (category, q) in enumerate(qs):
        passage, score = retriever.best_passage(q)
        context = passage if score >= args.min_context_score else None
        if context is not None:
            prompt = RAG_TEMPLATE.format(context=context, question=q)
        else:
            prompt = QA_TEMPLATE.format(question=q)
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        reply = "".join(chunk for chunk in stream_reply(
            model, idx, tokenizer, eot_id, args.length, args.temperature, args.top_k,
            args.repetition_penalty, no_penalty_ids, penalize_from=len(ids)))

        records.append({
            "category": category,
            "question": q,
            "score": score,
            "context_used": context is not None,
            "context": context,
            "answer": reply.strip(),
        })
        print(f"[gen {i+1}/{len(qs)}] {q}", flush=True)

    # Phase 2: judge everything except non_factual, concurrently
    num_to_judge = sum(1 for r in records if r["category"] != "non_factual")
    print(f"\njudging {num_to_judge} factual + false-premise answers ...", flush=True)
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as executor:
        records = list(executor.map(lambda r: classify(client, args.judge_model, r), records))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    tally = Counter(r["classification"] for r in records if "classification" in r)
    summary = {
        "run_name": run_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": args.checkpoint,
        "index_dir": args.index_dir or "default",
        "rerank": args.rerank,
        "min_context_score": args.min_context_score,
        "num_factual": args.num_factual,
        "num_factual_complex": args.num_factual_complex,
        "num_factual_open": args.num_factual_open,
        "num_false_premise": args.num_false_premise,
        "num_non_factual": args.num_non_factual,
        "tally": dict(tally),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nwrote {len(records)} records to {out_path}")
    print(f"wrote summary to {summary_path}\n")
    print("tally:")
    for label, count in tally.most_common():
        print(f"  {label:32s} {count}")


if __name__ == "__main__":
    main()
