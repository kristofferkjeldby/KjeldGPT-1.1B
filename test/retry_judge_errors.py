"""
Re-runs classify_closed_book only on records a prior rejudge_blackbox.py pass left as
"judge_error" (call_judge returned None -- an API/parse hiccup, not a real classification),
patching them into the existing <run_name>.jsonl / <run_name>_summary.json in place. Cheaper
and less noisy than a full rejudge_blackbox.py rerun, which would re-roll every record's
judgment and shift numbers that already resolved cleanly.

Run from within test/ (needs an ANTHROPIC_API_KEY):
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 retry_judge_errors.py --run_name v2_blackbox
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
    parser.add_argument("--run_name", type=str, required=True,
                         help="run_name to patch in place, e.g. v2_blackbox "
                              "(reads/writes runs/<run_name>.jsonl and _summary.json)")
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5")
    parser.add_argument("--judge_concurrency", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=3,
                         help="re-attempt records still judge_error after a pass, up to this many times")
    args = parser.parse_args()

    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    jsonl_path = os.path.join(runs_dir, f"{args.run_name}.jsonl")
    summary_path = os.path.join(runs_dir, f"{args.run_name}_summary.json")

    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    client = anthropic.Anthropic()
    total_retried = 0
    for attempt in range(args.max_retries):
        error_idxs = [i for i, r in enumerate(records) if r["classification"] == "judge_error"]
        if not error_idxs:
            break
        total_retried += len(error_idxs)
        print(f"attempt {attempt + 1}: retrying {len(error_idxs)} judge_error record(s)")
        with ThreadPoolExecutor(max_workers=args.judge_concurrency) as executor:
            fixed = list(executor.map(
                lambda i: classify_closed_book(client, args.judge_model, records[i]), error_idxs))
        for i, r in zip(error_idxs, fixed):
            records[i] = r

    remaining = sum(1 for r in records if r["classification"] == "judge_error")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    tally = {}
    for r in records:
        tally[r["classification"]] = tally.get(r["classification"], 0) + 1

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    summary["tally"] = tally
    summary["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"retried {total_retried} record-attempts, {remaining} still judge_error after {args.max_retries} passes")
    print("\ntally:")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()
