"""
Diagnoses the "rag_precision_failure" cases from a qa_loop.py run: is the wrong-context
problem a LOOKUP issue (a relevant passage exists in the corpus, just not ranked #1) or
a DATA issue (nothing in the corpus is actually relevant, or the embedding model can't
find it even broadening the search)?

For each rag_precision_failure question, re-searches with top_k=10 (instead of the 1
chat.py actually uses) and asks Claude to find the first genuinely relevant passage
among all 10 candidates, if any:

  lookup_issue          -- a relevant passage IS in the top-10, just not ranked #1 --
                           fixable by better ranking/reranking/embeddings, not more data
  coverage_or_embedding_gap -- nothing in the top-10 is relevant, but Wikipedia should
                           have this -- either the corpus lacks it, or the embedding
                           model can't semantically match this phrasing to it even
                           though it's there; needs manual follow-up to tell which
  no_coverage_expected  -- nothing in the top-10 is relevant, and it's a stretch to
                           expect Wikipedia to cleanly answer this specific question
                           anyway -- a hard/edge-case question, not really a fixable
                           retrieval failure
  rank1_reassessed_relevant -- this judge disagreed with qa_loop.py's original
                           classification and found rank 1 itself relevant after all
                           (rare judge-to-judge variance, not a real category)

Run from within test/ (needs ANTHROPIC_API_KEY):
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 diagnose_rag_precision.py --run_name v6
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from rag_retrieve import Retriever
from rag_rerank import Reranker

TOPK_JUDGE_PROMPT = """A retrieval system returned these {top_k} candidate passages for \
the question below, ranked by similarity score (rank 1 = the system's top match).

Question: {question}

Candidates:
{candidates}

Judge:
1. best_relevant_rank: the rank number (1-{top_k}) of the FIRST candidate that actually \
contains the specific information needed to answer the question -- not just topically \
related -- or null if none of them do.
2. wikipedia_should_have_this: even if none of the {top_k} candidates answer it, would a \
well-written Wikipedia article reasonably be expected to directly and clearly answer this \
question?

Respond with ONLY a JSON object: {{"best_relevant_rank": <int or null>, "wikipedia_should_have_this": true/false, "explanation": "one short sentence"}}"""


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
        return None


def classify(client, model, record, candidates):
    formatted = "\n\n".join(
        f"Rank {i+1} (score {score:.3f}): {passage[:400]}"
        for i, (passage, score) in enumerate(candidates)
    )
    judgment = call_judge(client, model, TOPK_JUDGE_PROMPT.format(
        top_k=len(candidates), question=record["question"], candidates=formatted))
    result = dict(record)
    result["topk_scores"] = [score for _, score in candidates]
    result["judgment"] = judgment
    if judgment is None:
        result["diagnosis"] = "judge_error"
    elif judgment.get("best_relevant_rank") == 1:
        result["diagnosis"] = "rank1_reassessed_relevant"
    elif judgment.get("best_relevant_rank"):
        result["diagnosis"] = "lookup_issue"
    elif judgment.get("wikipedia_should_have_this"):
        result["diagnosis"] = "coverage_or_embedding_gap"
    else:
        result["diagnosis"] = "no_coverage_expected"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True,
                         help="qa_loop.py run to pull cases from")
    parser.add_argument("--category", type=str, default="rag_precision_failure",
                         choices=["rag_precision_failure", "rag_recall_failure"],
                         help="which failure category to diagnose -- precision (context "
                              "retrieved but wrong) or recall (nothing scored above "
                              "threshold at all)")
    parser.add_argument("--index_dir", type=str, default=None,
                         help="passage index to re-search against -- defaults to "
                              "rag_retrieve.py's DEFAULT_INDEX_DIR; override only if the run "
                              "being diagnosed used some other index")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true",
                         help="rerank the top_k candidates with rag/rag_rerank.py's cross-encoder "
                              "before judging -- pass this when diagnosing a run that was "
                              "itself generated with qa_loop.py's --rerank, so the candidate "
                              "order/scores here match what that run actually saw")
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5")
    parser.add_argument("--judge_concurrency", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N cases (smoke test)")
    args = parser.parse_args()

    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    in_path = os.path.join(runs_dir, f"{args.run_name}.jsonl")
    out_path = os.path.join(runs_dir, f"{args.run_name}_{args.category}_diagnosis.jsonl")

    records = []
    with open(in_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("classification") == args.category:
                records.append(r)
    if args.limit:
        records = records[:args.limit]
    print(f"loaded {len(records)} {args.category} cases from {in_path}", flush=True)

    print("loading retriever ...", flush=True)
    retriever = Retriever(index_dir=args.index_dir) if args.index_dir else Retriever()
    reranker = Reranker() if args.rerank else None
    if reranker:
        print("loading cross-encoder reranker ...", flush=True)

    print(f"re-searching each with top_k={args.top_k}{' + reranking' if reranker else ''} ...", flush=True)
    candidates_by_question = {}
    for i, r in enumerate(records):
        candidates = retriever.top_passages(r["question"], top_k=args.top_k)
        if reranker:
            candidates = reranker.rerank(r["question"], candidates)
        candidates_by_question[r["question"]] = candidates
        print(f"  [{i+1}/{len(records)}] {r['question']}", flush=True)

    print(f"\njudging {len(records)} cases ...", flush=True)
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as executor:
        results = list(executor.map(
            lambda r: classify(client, args.judge_model, r, candidates_by_question[r["question"]]),
            records))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    tally = Counter(r["diagnosis"] for r in results)
    print(f"\nwrote {len(results)} records to {out_path}\n")
    print("diagnosis tally:")
    for label, count in tally.most_common():
        print(f"  {label:32s} {count}")


if __name__ == "__main__":
    main()
