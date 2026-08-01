"""
Parses train_run.log's [eval]/[console] lines and (re)generates the loss/throughput
plots under insights/plots/ (both alongside this script, in insights/).

Run from anywhere -- --log/--out_dir default to logs/ and plots/ next to this script
(base/insights/), not the current working directory:
    python3 base/insights/plot_insights.py
    python3 base/insights/plot_insights.py --log /path/to/train_run.log --out_dir /path/to/plots

Note: the "Chinchilla predicted floor" line is computed from the Hoffmann et al. 2022
("Training Compute-Optimal Large Language Models") approach-3 scaling law,
L(N,D) = E + A/N^alpha + B/D^beta, plugged in with this model's actual parameter count
and its full planned token budget (MAX_ITERS * tokens/step) -- currently ~2.57, consistent
with this script's own separately-fitted power-law extrapolation.
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit

# model.py lives at the repo root, two levels up from base/insights/ (this script's
# own directory, regardless of the caller's cwd).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model import GPT, GPTConfig

# Hoffmann et al. 2022 ("Training Compute-Optimal Large Language Models") approach-3
# scaling-law fit constants -- see docstring.
CHINCHILLA_E, CHINCHILLA_A, CHINCHILLA_B = 1.69, 406.4, 410.7
CHINCHILLA_ALPHA, CHINCHILLA_BETA = 0.34, 0.28

# Must match base_train.py's hyperparameters -- these determine tokens/step and the
# total planned training budget used for the fit's extrapolation target.
BATCH_SIZE = 24
BLOCK_SIZE = 1024
MAX_ITERS = 808998
N_LAYER, N_HEAD, N_EMBD, VOCAB_SIZE = 36, 24, 1536, 50257


def chinchilla_floor(n_params, tokens):
    return (CHINCHILLA_E + CHINCHILLA_A / n_params ** CHINCHILLA_ALPHA
            + CHINCHILLA_B / tokens ** CHINCHILLA_BETA)

EVAL_RE = re.compile(
    r"\[eval\] step\s+(\d+)/\d+ \| train loss ([\d.]+) .*? \| val loss ([\d.]+) .*?\| "
    r"best val ([\d.]+)@\d+ .*?\| ([\d,]+) tok/s"
)
CONSOLE_RE = re.compile(r"\[console\] step\s+(\d+)/\d+ .*?\| ([\d,]+) tok/s")


def param_count():
    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=N_LAYER,
                        n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0, tied=True)
    return sum(p.numel() for p in GPT(config).parameters())


def parse_log(path):
    evals = []  # (step, train_loss, val_loss, best_val, tok_s)
    throughput = []  # (step, tok_s)
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                step, train_loss, val_loss, best_val, tok_s = m.groups()
                evals.append((int(step), float(train_loss), float(val_loss), float(best_val),
                              float(tok_s.replace(",", ""))))
                throughput.append((int(step), float(tok_s.replace(",", ""))))
                continue
            m = CONSOLE_RE.search(line)
            if m:
                step, tok_s = m.groups()
                throughput.append((int(step), float(tok_s.replace(",", ""))))
    evals.sort(key=lambda r: r[0])
    throughput.sort(key=lambda r: r[0])
    return evals, throughput


def power_law(t, l_inf, a, alpha):
    return l_inf + a * np.power(t, -alpha)


def fit_power_law(tokens, val_loss):
    # Unbounded, l_inf/A/alpha are only ~28x-of-token-range apart in this run's eval
    # points -- too narrow to identify all three independently, so an unconstrained
    # curve_fit finds a numerically-cancelling (l_inf very negative, alpha near zero)
    # combination that fits the visible range but is meaningless extrapolated to
    # max_iters (confirmed via param std errors ~5 orders of magnitude past the fitted
    # values themselves). Bounding l_inf below the observed loss floor and alpha to a
    # normal power-law decay range keeps the fit physically identifiable.
    popt, _ = curve_fit(power_law, tokens, val_loss, p0=[2.3, 1.0, 0.3],
                         bounds=([0.5, 1e-6, 0.01], [val_loss.min(), 1e6, 3.0]), maxfev=50000)
    return popt  # l_inf, a, alpha


def plot_loss_vs_step(evals, floor, out_path):
    steps = [e[0] for e in evals]
    train_loss = [e[1] for e in evals]
    val_loss = [e[2] for e in evals]
    best_val = [e[3] for e in evals]

    plt.figure(figsize=(9, 5.2))
    plt.plot(steps, train_loss, color="tab:blue", alpha=0.6, linewidth=1, label="train loss (eval-time)")
    plt.plot(steps, val_loss, color="tab:orange", linewidth=1.8, label="val loss")
    plt.plot(steps, best_val, color="tab:green", linestyle="--", linewidth=1.5, label="best val (so far)")
    plt.axhline(floor, color="gray", linestyle=":", linewidth=1,
                label=f"Chinchilla predicted floor ({floor:.3f})")
    plt.xlabel("training step")
    plt.ylabel("loss (nats/token)")
    plt.title("KjeldGPT 1.1B — training/val loss vs step")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_loss_vs_x(x, val_loss, fit_x, fit_y, final_x, final_y, xlabel, title, out_path, l_inf, a, alpha, floor):
    # Subtracting l_inf turns the power law L = l_inf + a*t^-alpha into a*t^-alpha,
    # which is a straight line once both axes are log-scaled -- the raw loss on a
    # semi-log axis is a curve, not a line, since it's decaying towards an offset.
    plt.figure(figsize=(9, 5.2))
    plt.scatter(x, val_loss - l_inf, color="tab:orange", s=14, label="val loss (eval points)")
    plt.plot(fit_x, fit_y - l_inf, color="tab:blue", linewidth=1.5,
              label=f"power-law fit: L−{l_inf:.3f}={a:.2f}·t$^{{-{alpha:.3f}}}$")
    plt.scatter([final_x], [final_y - l_inf], color="tab:blue", marker="*", s=180,
                label=f"fit-predicted final loss ({final_y:.3f})", zorder=5)
    plt.axvline(final_x, color="gray", linestyle="--", linewidth=1)
    if floor > l_inf:
        plt.axhline(floor - l_inf, color="gray", linestyle=":", linewidth=1,
                    label=f"Chinchilla predicted floor ({floor:.3f})")
    plt.xscale("log")
    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    plt.xlabel(xlabel)
    plt.ylabel("val loss (nats/token, log scale)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_throughput_vs_step(throughput, out_path):
    steps = [t[0] for t in throughput]
    tok_s = [t[1] for t in throughput]
    plt.figure(figsize=(9, 4))
    plt.plot(steps, tok_s, color="mediumpurple", linewidth=1.3)
    plt.xlabel("training step")
    plt.ylabel("tokens/sec")
    plt.title("KjeldGPT 1.1B — throughput vs step")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "train_run.log"))
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
    args = parser.parse_args()

    evals, throughput = parse_log(args.log)
    if not evals:
        raise SystemExit(f"no [eval] lines found in {args.log}")
    print(f"parsed {len(evals)} eval points, {len(throughput)} throughput points from {args.log}")

    n_params = param_count()
    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    tokens = np.array([e[0] * tokens_per_step for e in evals], dtype=float)
    val_loss = np.array([e[2] for e in evals], dtype=float)
    flops = 6 * n_params * tokens

    l_inf, a, alpha = fit_power_law(tokens, val_loss)
    final_tokens = MAX_ITERS * tokens_per_step
    final_flops = 6 * n_params * final_tokens
    final_loss = power_law(final_tokens, l_inf, a, alpha)
    print(f"power-law fit: L_inf={l_inf:.3f} A={a:.2f} alpha={alpha:.3f} -- "
          f"predicted final loss {final_loss:.3f} at {final_tokens:,.0f} tokens")

    floor = chinchilla_floor(n_params, final_tokens)
    print(f"Chinchilla predicted floor: {floor:.3f} (N={n_params:,.0f} params, "
          f"D={final_tokens:,.0f} planned tokens)")

    os.makedirs(args.out_dir, exist_ok=True)

    plot_loss_vs_step(evals, floor, os.path.join(args.out_dir, "loss_vs_step.png"))

    fit_tokens = np.geomspace(tokens.min(), final_tokens, 200)
    fit_loss = power_law(fit_tokens, l_inf, a, alpha)
    plot_loss_vs_x(
        tokens, val_loss, fit_tokens, fit_loss, final_tokens, final_loss,
        "tokens processed (log scale)", "KjeldGPT 1.1B — val loss vs tokens (log-log), with power-law extrapolation",
        os.path.join(args.out_dir, "loss_vs_tokens_loglog.png"), l_inf, a, alpha, floor,
    )
    fit_flops = 6 * n_params * fit_tokens
    plot_loss_vs_x(
        flops, val_loss, fit_flops, fit_loss, final_flops, final_loss,
        "training compute, FLOPs = 6·N·tokens (log scale)",
        "KjeldGPT 1.1B — val loss vs compute (log-log)",
        os.path.join(args.out_dir, "loss_vs_flops_loglog.png"), l_inf, a, alpha, floor,
    )

    plot_throughput_vs_step(throughput, os.path.join(args.out_dir, "throughput_vs_step.png"))

    print(f"wrote 4 plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
