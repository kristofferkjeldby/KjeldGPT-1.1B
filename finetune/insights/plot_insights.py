"""
Parses finetune_run.log's [eval]/[console] lines and generates loss/throughput plots
under finetune/insights/plots/ (both alongside this script) -- parallel to
../../insights/plot_insights.py, but simpler: no power-law/Chinchilla extrapolation,
since this run is a fixed 3-epoch budget over a tiny corpus, not a smooth scaling-law
trajectory worth fitting or extrapolating.

Run from anywhere -- --log/--out_dir default to logs/ and plots/ next to this script,
not the current working directory:
    python3 finetune/insights/plot_insights.py
    python3 finetune/insights/plot_insights.py --log /path/to/finetune_run.log --out_dir /path/to/plots
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_RE = re.compile(
    r"\[eval\] step\s+(\d+)/\d+ \| train loss ([\d.]+) .*? \| val loss ([\d.]+) .*?\| "
    r"best val ([\d.]+)@\d+ .*?\| ([\d,]+) tok/s"
)
CONSOLE_RE = re.compile(r"\[console\] step\s+(\d+)/\d+ .*?\| ([\d,]+) tok/s")


def parse_log(path):
    evals = []
    throughput = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                step, train_loss, val_loss, best_val, tok_s = m.groups()
                evals.append((int(step), float(train_loss), float(val_loss), float(best_val)))
                throughput.append((int(step), float(tok_s.replace(",", ""))))
                continue
            m = CONSOLE_RE.search(line)
            if m:
                step, tok_s = m.groups()
                throughput.append((int(step), float(tok_s.replace(",", ""))))
    evals.sort(key=lambda r: r[0])
    throughput.sort(key=lambda r: r[0])
    return evals, throughput


def plot_loss_vs_step(evals, out_path):
    steps = [e[0] for e in evals]
    train_loss = [e[1] for e in evals]
    val_loss = [e[2] for e in evals]
    best_val = [e[3] for e in evals]
    best_step = min(evals, key=lambda e: e[2])[0]
    best_loss = min(e[2] for e in evals)

    plt.figure(figsize=(9, 5.2))
    plt.plot(steps, train_loss, color="tab:blue", alpha=0.6, linewidth=1.2, marker="o", markersize=3,
              label="train loss (eval-time)")
    plt.plot(steps, val_loss, color="tab:orange", linewidth=1.8, marker="o", markersize=3, label="val loss")
    plt.plot(steps, best_val, color="tab:green", linestyle="--", linewidth=1.5, label="best val (so far)")
    plt.scatter([best_step], [best_loss], color="tab:green", marker="*", s=200, zorder=5,
                label=f"best checkpoint (step {best_step}, {best_loss:.4f})")
    plt.xlabel("training step")
    plt.ylabel("loss (nats/token)")
    plt.title("KjeldChat 1.1B — finetuning loss vs step")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_throughput_vs_step(throughput, out_path):
    steps = [t[0] for t in throughput]
    tok_s = [t[1] for t in throughput]
    plt.figure(figsize=(9, 4))
    plt.plot(steps, tok_s, color="mediumpurple", linewidth=1.3, marker="o", markersize=3)
    plt.xlabel("training step")
    plt.ylabel("tokens/sec")
    plt.title("KjeldChat 1.1B — finetuning throughput vs step")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "finetune_run.log"))
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
    args = parser.parse_args()

    evals, throughput = parse_log(args.log)
    if not evals:
        raise SystemExit(f"no [eval] lines found in {args.log}")
    print(f"parsed {len(evals)} eval points, {len(throughput)} throughput points from {args.log}")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_loss_vs_step(evals, os.path.join(args.out_dir, "loss_vs_step.png"))
    plot_throughput_vs_step(throughput, os.path.join(args.out_dir, "throughput_vs_step.png"))
    print(f"wrote 2 plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
