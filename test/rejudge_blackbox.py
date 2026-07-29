"""
Re-judges an existing whitebox qa_loop.py run (e.g. v6.jsonl) closed-book, blackbox
style -- for the KjeldChat vs. external-model comparison chart (plot_model_comparison.py),
where KjeldChat needs to be graded on the same "question in, answer out" basis as the
external models, ignoring the RAG machinery (context_used/context/score) entirely. See
qa_loop_external.py's classify_closed_book for the actual judging logic -- this script
just feeds it KjeldChat's own already-generated answers instead of calling an API/local
model, since those answers already exist in the source run's .jsonl.

Run from within test/ (needs an ANTHROPIC_API_KEY for the judge phase):
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 rejudge_blackbox.py --source v6
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

from qa_loop_external import classify_closed_book


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                         help="run_name of the existing whitebox run to re-judge, e.g. v6 "
                              "(reads runs/<source>.jsonl)")
    parser.add_argument("--run_name", type=str, default="KjeldChat",
                         help="output run_name -- writes runs/<run_name>.jsonl and "
                              "runs/<run_name>_summary.json")
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5")
    parser.add_argument("--judge_concurrency", type=int, default=5)
    args = parser.parse_args()

    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    source_path = os.path.join(runs_dir, f"{args.source}.jsonl")

    records = []
    with open(source_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for key in ("score", "context_used", "context", "classification", "judgment"):
                rec.pop(key, None)
            records.append(rec)

    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as executor:
        records = list(executor.map(
            lambda r: classify_closed_book(client, args.judge_model, r), records))

    out_path = os.path.join(runs_dir, f"{args.run_name}.jsonl")
    summary_path = os.path.join(runs_dir, f"{args.run_name}_summary.json")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    tally = {}
    for r in records:
        tally[r["classification"]] = tally.get(r["classification"], 0) + 1

    summary = {
        "run_name": args.run_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": f"blackbox re-judge of {args.source!r} (../finetune/checkpoints/kjeldchat_v6.pt)",
        "num_factual": sum(1 for r in records if r["category"] == "factual"),
        "num_false_premise": sum(1 for r in records if r["category"] == "false_premise"),
        "num_non_factual": sum(1 for r in records if r["category"] == "non_factual"),
        "tally": tally,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {len(records)} records to {out_path}")
    print(f"wrote summary to {summary_path}")
    print("\ntally:")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()
