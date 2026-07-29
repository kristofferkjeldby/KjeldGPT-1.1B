"""
Stacked bar chart comparing KjeldChat (finetune + RAG, blackbox re-judged -- see
rejudge_blackbox.py) against every external/hobby-scale model tested via
qa_loop_external.py, all on identical closed-book footing: same 426-question set, same
correctness-only judge (qa_loop_external.classify_closed_book), no model given retrieval
context it didn't build itself. This is a genuine blackbox comparison -- KjeldChat's own
internal RAG mechanism is invisible here too, exactly as an external tester evaluating 7
unknown systems would see it (only question in, answer out).

Run from within test/, once the runs it plots exist in runs/ (KjeldChat_summary.json via
rejudge_blackbox.py, the rest via qa_loop_external.py):
    python3 plot_model_comparison.py
    python3 plot_model_comparison.py --runs KjeldChat gpt-3.5-turbo davinci-002
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same status-severity palette as plot_qa_loop.py's CATEGORY_COLOR, extended with
# "incorrect" -- the single collapsed failure bucket every closed-book run produces,
# replacing the whitebox-only rag_recall_failure/rag_precision_failure/
# finetuning_grounding_failure split that requires knowing how a system is built inside.
CATEGORY_ORDER = [
    "success",
    "premise_corrected",
    "non_factual",
    "incorrect",
    "premise_accepted_hallucination",
    "judge_error",
]
CATEGORY_COLOR = {
    "success": "#0ca30c",                  # good
    "premise_corrected": "#0b5e1c",        # dark green -- also a success, but a harder-won
                                            # one (catching a false premise rather than
                                            # just retrieving a fact), worth distinguishing
    "non_factual": "#898781",              # muted -- not evaluated (hallucination OK)
    "incorrect": "#d03b3b",                # critical -- wrong answer, closed-book
    "premise_accepted_hallucination": "#4a3aa7",  # violet -- confidently hallucinated
                                            # a fact for a false-premise question
    "judge_error": "#c3c2b7",              # rare: judge response didn't parse
}
CATEGORY_LABEL = {
    "success": "success",
    "premise_corrected": "premise\ncorrected",
    "non_factual": "non-factual\n(OK)",
    "incorrect": "incorrect",
    "premise_accepted_hallucination": "false premise\naccepted",
    "judge_error": "judge\nerror",
}

DEFAULT_RUNS = [
    "KjeldChat", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct",
    "davinci-002", "babbage-002", "tinyllama-1.1b-chat", "gpt2-xl",
]
DEFAULT_LABELS = {
    "KjeldChat": "KjeldChat\n(finetune + RAG)",
    "gpt-3.5-turbo": "GPT-3.5\nturbo",
    "gpt-3.5-turbo-instruct": "GPT-3.5\nturbo-instruct",
    "davinci-002": "davinci-002",
    "babbage-002": "babbage-002",
    "tinyllama-1.1b-chat": "TinyLlama\n1.1B-Chat",
    "gpt2-xl": "GPT-2 XL\n(1.5B)",
}


def load_summary(runs_dir, name):
    with open(os.path.join(runs_dir, f"{name}_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
    parser.add_argument("--runs", type=str, nargs="+", default=DEFAULT_RUNS)
    args = parser.parse_args()

    pairs = [(name, load_summary(args.runs_dir, name)) for name in args.runs]
    # Worst to best, left to right -- ranked by success count rather than a fixed
    # hardcoded order, so this stays correct if a run is redone or a new one is added.
    pairs.sort(key=lambda p: p[1]["tally"].get("success", 0))
    args.runs = [name for name, _ in pairs]
    summaries = [summary for _, summary in pairs]
    labels = [DEFAULT_LABELS.get(name, name) for name in args.runs]
    os.makedirs(args.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = list(range(len(summaries)))
    bottoms = [0] * len(summaries)
    for cat in CATEGORY_ORDER:
        heights = [s["tally"].get(cat, 0) for s in summaries]
        if not any(heights):
            continue
        bars = ax.bar(x, heights, bottom=bottoms, color=CATEGORY_COLOR[cat],
                       label=CATEGORY_LABEL[cat].replace("\n", " "), width=0.6)
        for bar, h, b in zip(bars, heights, bottoms):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, b + h / 2, str(h),
                        ha="center", va="center", fontsize=8, color="white")
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("questions (of 426)")
    ax.set_title(
        "Blackbox comparison -- KjeldChat vs. external/hobby-scale models\n"
        "same 426 questions"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "model_comparison.png")
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
