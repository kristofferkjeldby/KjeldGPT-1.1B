"""
Plots rejudge_blackbox.py's category tally as a trend line across runs -- the closed-book,
real-world-correctness counterpart to plot_qa_loop.py's whitebox trend. Same visual
convention (fixed status-severity palette, movement-between-runs is what matters), but
over the blackbox tally's smaller category set (no rag_*/finetuning_grounding_failure --
those are whitebox-only, since blackbox judging never sees what Context was retrieved).

Run from within test/ once there are 2+ *_blackbox_summary.json files in runs/:
    python3 plot_blackbox_trend.py --runs v1_blackbox v2_blackbox v3_blackbox v4_blackbox v5_blackbox v6_blackbox KjeldChat --labels v1 v2 v3 v4 v5 v6 v7
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORY_ORDER = [
    "success",
    "premise_corrected",
    "non_factual",
    "premise_accepted_hallucination",
    "incorrect",
    "judge_error",
]
CATEGORY_COLOR = {
    "success": "#0ca30c",                         # good
    "premise_corrected": "#0b5e1c",               # dark green -- a harder-won success
    "non_factual": "#898781",                     # muted -- not evaluated (hallucination OK)
    "premise_accepted_hallucination": "#4a3aa7",  # violet -- confidently hallucinated over a false premise
    "incorrect": "#d03b3b",                       # critical -- wrong, real-world
    "judge_error": "#c3c2b7",                     # rare: judge response didn't parse
}
CATEGORY_LABEL = {
    "success": "success",
    "premise_corrected": "premise\ncorrected",
    "non_factual": "non-factual\n(OK)",
    "premise_accepted_hallucination": "false premise\naccepted",
    "incorrect": "incorrect",
    "judge_error": "judge\nerror",
}


def load_runs(runs_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*_summary.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    runs.sort(key=lambda r: r["timestamp_utc"])
    return runs


def plot_trend(runs, out_path, labels=None):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = list(range(len(runs)))
    for cat in CATEGORY_ORDER:
        ys = [r["tally"].get(cat, 0) for r in runs]
        if not any(ys):
            continue
        ax.plot(x, ys, marker="o", markersize=6, linewidth=2, color=CATEGORY_COLOR[cat],
                label=CATEGORY_LABEL[cat].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(labels or [r["run_name"] for r in runs], rotation=20, ha="right")
    ax.set_ylabel("questions")
    ax.set_title("Test results v1 to v7")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                         help="explicit run_names to plot, in this order")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                         help="display labels for --runs (same order/length)")
    args = parser.parse_args()

    all_runs = load_runs(args.runs_dir)
    if not all_runs:
        raise SystemExit(f"no *_summary.json files found in {args.runs_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.runs:
        by_name = {r["run_name"]: r for r in all_runs}
        missing = [name for name in args.runs if name not in by_name]
        if missing:
            raise SystemExit(f"run(s) not found in {args.runs_dir}: {missing}")
        selected = [by_name[name] for name in args.runs]
        if args.labels and len(args.labels) != len(selected):
            raise SystemExit(f"--labels has {len(args.labels)} entries, expected {len(selected)}")
        labels = args.labels or [r["run_name"] for r in selected]
    else:
        selected, labels = all_runs, None

    if len(selected) < 2:
        raise SystemExit("trend plot needs 2+ runs")

    trend_path = os.path.join(args.out_dir, "blackbox_trend.png")
    plot_trend(selected, trend_path, labels=labels)
    print(f"wrote {trend_path} ({len(selected)} runs)")


if __name__ == "__main__":
    main()
