"""
Plots qa_loop.py's category tally as a trend line per category across runs, so
progress (or regression) across successive fixes is visible at a glance rather
than buried in JSON. The per-run snapshot is deliberately not plotted -- a single
run's tally is already legible as JSON in runs/<name>_summary.json, and what
actually carries information is the movement between runs.

Categories are colored by outcome severity (a fixed status palette, not an arbitrary
categorical one), since they're inherently good/bad states, not unordered identities:
success (good), the two "not a real failure" categories (muted neutral), and the
three failure categories (warning/serious/critical, ordered by how directly each
implicates finetuning vs. retrieval).

Run from within test/ once there are 2+ qa_loop.py runs:
    python3 plot_qa_loop.py
    python3 plot_qa_loop.py --runs v1 v2 v3 v4 v5 v6
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed order + status-palette colors --
# color reinforces severity, not identity, since each category is already labeled on
# its own axis tick.
CATEGORY_ORDER = [
    "success",
    "premise_corrected",
    "non_factual",
    "no_context_expected",
    "rag_recall_failure",
    "rag_precision_failure",
    "finetuning_grounding_failure",
    "premise_accepted_hallucination",
    "judge_error",
]
CATEGORY_COLOR = {
    "success": "#0ca30c",                  # good
    "premise_corrected": "#0b5e1c",        # dark green -- a harder-won success than plain
                                            # correctness (see plot_model_comparison.py)
    "non_factual": "#898781",              # muted -- not evaluated (hallucination OK)
    "no_context_expected": "#898781",      # muted -- not a failure, question isn't wikipedia-answerable
    "rag_recall_failure": "#fab219",       # warning -- retrieval missed an available passage
    "rag_precision_failure": "#ec835a",    # serious -- retrieved passage wasn't relevant
    "finetuning_grounding_failure": "#d03b3b",  # critical -- relevant context, not used correctly
    "premise_accepted_hallucination": "#4a3aa7",  # violet -- distinct failure mode: confidently
                                            # hallucinated a fact for a question with a false premise,
                                            # not a RAG or ordinary grounding issue
    "judge_error": "#c3c2b7",              # rare: judge response didn't parse
}
CATEGORY_LABEL = {
    "success": "success",
    "premise_corrected": "premise\ncorrected",
    "non_factual": "non-factual\n(OK)",
    "no_context_expected": "no context\nexpected",
    "rag_recall_failure": "RAG recall\nfailure",
    "rag_precision_failure": "RAG precision\nfailure",
    "finetuning_grounding_failure": "finetuning\ngrounding failure",
    "premise_accepted_hallucination": "false premise\naccepted",
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
    ax.set_title("QA loop -- category counts across runs")
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
                         help="explicit run_names to plot, in this order -- for restricting "
                              "the trend to a chosen subset (e.g. just the whitebox v-runs, "
                              "skipping the blackbox re-judge). Omit to plot every run in "
                              "runs_dir, ordered by timestamp.")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                         help="display labels for --runs (same order/length) -- overrides "
                              "each run's run_name on the x-axis, for presenting a cleaner "
                              "name than the run's recorded identifier")
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
        raise SystemExit("trend plot needs 2+ runs -- a single run's tally is already "
                          "readable in its runs/<name>_summary.json")

    trend_path = os.path.join(args.out_dir, "qa_loop_trend.png")
    plot_trend(selected, trend_path, labels=labels)
    print(f"wrote {trend_path} ({len(selected)} runs)")


if __name__ == "__main__":
    main()
