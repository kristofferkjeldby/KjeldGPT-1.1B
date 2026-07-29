"""
Finetunes the pretrained GPT from model.py on the Q/A corpus -- parallel to
base_train.py, but adjusted for finetuning: a much lower learning rate (continuing
training on a tiny, narrow corpus at the pretrain peak_lr would blow away what the base
model already learned), a short warmup sized to a run of a few hundred steps rather than
hundreds of thousands, step-based (not time-based) eval/checkpoint/console cadences
since a full run here takes minutes, not days, and masked loss -- the target at every
prompt-token position is -100, which model.py's cross_entropy(ignore_index=-100) skips,
so the model trains on producing the answer, not reproducing the question.

Run tokenize_finetune.py first (after shuffle_finetune.py). Intended to run from the same directory
as its data, mirroring base_train.py's convention:
    cd finetune
    python3 finetune_train.py --dropout 0.2 --peak_lr 1e-5 --patience 10

That is the settled recipe -- every run since Round 4 uses exactly those flags, so runs
differ only by their corpus and stay comparable. See ../FINETUNE_PARAMS.md.

Writes two checkpoints into --out_dir: kjeldchat.pt (latest, on the periodic cadence)
and kjeldchat_best.pt (every time val loss improves). The latter is the one to keep.
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

# model.py lives at the repo root, but a remote training box may just copy it flat
# alongside this script -- try both locations.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true",
                     help="print gradient norms and weight-nudge magnitudes at every eval")
parser.add_argument("--patience", type=int, default=5,
                     help="stop after this many consecutive evals with no val loss "
                          "improvement -- lower than base_train.py's default since a "
                          "finetune run has far fewer evals total")
parser.add_argument("--resume", type=str, default="../base/checkpoints/kjeldgpt.pt",
                     help="pretrained base checkpoint to finetune from")
parser.add_argument("--peak_lr", type=float, default=None, help="override the peak learning rate below")
parser.add_argument("--dropout", type=float, default=None, help="override the dropout rate below")
parser.add_argument("--out_dir", type=str, default=None, help="override the checkpoint save directory below")
parser.add_argument("--epochs", type=float, default=3.0,
                     help="target passes over the finetune corpus -- max_iters is "
                          "derived from this and the actual token count in meta.json, "
                          "since this corpus's size is expected to keep changing")
parser.add_argument("--batch_size", type=int, default=8,
                     help="lower than base_train.py's 24 -- with only a few hundred "
                          "total steps available, smaller batches trade some throughput "
                          "for more optimizer steps to actually use the schedule")
args = parser.parse_args()
DEBUG = args.debug

# ---- hyperparameters -------------------------------------------------------
# Architecture must match the checkpoint being resumed from exactly (block_size in
# particular -- the pretrained pos_emb table is sized 1024 and can't be extended).
batch_size = args.batch_size
block_size = 1024
n_layer = 36
n_head = 24
n_embd = 1536
dropout = args.dropout if args.dropout is not None else 0.1
                         # bumped from the base run's 0.05 -- this corpus is a few
                         # hundred thousand tokens against a 1.1B-param model, so
                         # overfitting across only a handful of epochs is a real risk
                         # in a way it wasn't for the 6.6B-token pretrain run. Dropout
                         # has no learnable parameters, so changing it doesn't affect
                         # whether the checkpoint's weights load.
tied = True              # must match the resumed checkpoint

warmup_iters = 10        # a full run here is a few hundred steps, not 808,998 -- scale
                         # the warmup window down to match, same proportion roughly
peak_lr = args.peak_lr if args.peak_lr is not None else 2e-5
                         # an order of magnitude below the base run's 2e-4 peak --
                         # standard finetuning practice: a narrow, tiny corpus at the
                         # pretrain LR would overwrite general capability learned over
                         # 6.6B tokens in a few hundred steps instead of adapting it
min_lr = peak_lr / 10
weight_decay = 0.1

eval_every = 10          # steps, not seconds -- time-based cadences (base_train.py's
                         # 300s/1800s/3600s) assume a multi-day run; this one finishes
                         # in minutes
checkpoint_every = 20
console_every = 5
eval_iters = 8           # train-loss readout only (see estimate_loss) -- a rough,
                         # noisy training-loss diagnostic is fine since it doesn't
                         # drive any decision. Val loss is handled separately (see
                         # make_val_windows/estimate_loss): early stopping needs
                         # consecutive evals to be comparable, so val loss is measured
                         # over a fixed, full, deterministic pass over the val set
                         # rather than a random resample each time.

out_dir = args.out_dir if args.out_dir is not None else "checkpoints"
data_dir = "data"
# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" \
    else contextlib.nullcontext()


def get_lr(it, max_iters):
    if it < warmup_iters:
        return peak_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def load_data():
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.dtype(meta["dtype"])
    train_ids = np.memmap(os.path.join(data_dir, "train_ids.bin"), dtype=dtype, mode="r")
    train_mask = np.memmap(os.path.join(data_dir, "train_mask.bin"), dtype=np.uint8, mode="r")
    val_ids = np.memmap(os.path.join(data_dir, "val_ids.bin"), dtype=dtype, mode="r")
    val_mask = np.memmap(os.path.join(data_dir, "val_mask.bin"), dtype=np.uint8, mode="r")
    return train_ids, train_mask, val_ids, val_mask, meta


def get_batch(split, data, ix=None):
    """ix=None (training steps, and estimate_loss's train-loss readout): batch_size
    random window starts, a fresh draw every call -- fine for both, since neither
    drives a decision that requires two calls to be comparable to each other.
    ix=<explicit list> (estimate_loss's val pass): reuses make_val_windows' fixed
    offsets instead of drawing new random ones, so every eval measures the exact same
    val windows as every other eval."""
    train_ids, train_mask, val_ids, val_mask = data
    ids, mask = (train_ids, train_mask) if split == "train" else (val_ids, val_mask)
    if ix is None:
        ix = torch.randint(len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(ids[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(ids[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    keep = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + block_size].astype(bool)) for i in ix])
    # Prompt-token targets become -100 -- model.py's cross_entropy(ignore_index=-100)
    # excludes them from the loss entirely, so only answer tokens (and the trailing
    # EOT) drive the gradient.
    y = torch.where(keep, y, torch.full_like(y, -100))
    return x.to(device), y.to(device)


def make_val_windows(val_ids):
    """Fixed, non-overlapping window start offsets spanning the entire val set,
    computed once at startup. Passed to estimate_loss so every eval scores the exact
    same val windows in the exact same order -- unlike get_batch's random sampling
    (fine for training steps, where each draw is independent and nothing needs to
    compare across steps), val loss specifically drives the early-stopping decision,
    so it needs to be a stable, apples-to-apples signal from one eval to the next
    rather than a fresh random resample each time."""
    last_valid_start = len(val_ids) - block_size - 1
    return list(range(0, last_valid_start + 1, block_size))


def grad_norm(params):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None
    return torch.sqrt(sum((g ** 2).sum() for g in grads)).item()


@torch.no_grad()
def estimate_loss(model, data, val_windows):
    out = {}
    model.eval()

    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch("train", data)
        with autocast_ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    out["train"] = losses.mean().item()

    # Full, fixed, deterministic pass over the val set (see make_val_windows) --
    # every eval covers exactly the same windows, so val loss is actually comparable
    # step to step instead of a fresh random resample each time.
    losses = []
    for start in range(0, len(val_windows), batch_size):
        ix = val_windows[start:start + batch_size]
        x, y = get_batch("val", data, ix=ix)
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    out["val"] = sum(losses) / len(losses)

    model.train()
    return out


def main():
    print(f"device: {device}  patience: {args.patience} evals")
    train_ids, train_mask, val_ids, val_mask, meta = load_data()
    data = (train_ids, train_mask, val_ids, val_mask)
    val_windows = make_val_windows(val_ids)
    print(f"val eval: {len(val_windows)} fixed windows spanning the full val set "
          f"({len(val_windows) * block_size:,} of {meta['val_tokens']:,} val tokens), "
          f"same windows scored every eval")
    vocab_size = meta["vocab_size"]
    tokens_per_step = batch_size * block_size
    max_iters = max(1, int(args.epochs * meta["train_tokens"] // tokens_per_step))
    print(f"data: {meta['train_tokens']:,} train tokens ({meta['num_pairs'] - meta['num_val_pairs']:,} pairs), "
          f"{meta['val_tokens']:,} val tokens ({meta['num_val_pairs']:,} pairs), vocab_size={vocab_size}")
    print(f"hyperparams: n_layer={n_layer} n_head={n_head} n_embd={n_embd} block_size={block_size} "
          f"batch_size={batch_size} tied={tied} weight_decay={weight_decay} dropout={dropout}")
    print(f"lr schedule: warmup_iters={warmup_iters} peak_lr={peak_lr:.2e} min_lr={min_lr:.2e} "
          f"max_iters={max_iters} ({args.epochs} epochs)")
    print(f"console every {console_every} steps, eval every {eval_every} steps, "
          f"checkpoint every {checkpoint_every} steps ({tokens_per_step:,} tokens/step)")

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        tied=tied,
    )
    model = GPT(config).to(device)
    model_ties_weights = model.head.weight is model.tok_emb.weight

    start_it = 0
    best_val_loss = float("inf")
    best_iter = 0
    if not args.resume:
        raise SystemExit("finetuning requires --resume pointing at a pretrained base checkpoint")

    torch.serialization.add_safe_globals([GPTConfig])
    ckpt = torch.load(args.resume, map_location=device)
    raw_sd = ckpt["model"]
    # Same tied-weight handling as base_train.py's --resume: only relevant if the
    # checkpoint predates weight tying, kept for consistency/robustness rather than
    # because this particular base checkpoint needs it (it doesn't -- tied=True already).
    architecture_changed = (model_ties_weights
                             and "head.weight" in raw_sd
                             and not torch.equal(raw_sd["tok_emb.weight"], raw_sd["head.weight"]))
    if architecture_changed:
        averaged = (raw_sd["tok_emb.weight"] + raw_sd["head.weight"]) / 2
        raw_sd = dict(raw_sd)
        raw_sd["tok_emb.weight"] = averaged
        del raw_sd["head.weight"]
        print("resume: checkpoint predates weight tying -- averaging tok_emb.weight/head.weight")
    model.load_state_dict(raw_sd, strict=False)
    print(f"resumed base checkpoint from {args.resume}: step {ckpt['iter']}, val_loss {ckpt['val_loss']:.4f} "
          f"-- starting a fresh finetune run at step 0 with its own best-val tracking")
    # start_it/best_val_loss/best_iter intentionally reset to 0/inf here (unlike
    # base_train.py's --resume): this is a new training objective (masked QA loss),
    # not a continuation of the same one, so the base checkpoint's step count and val
    # loss aren't comparable to this run's.

    # foreach=False: PyTorch's default fused multi-tensor Adam path (torch._foreach_sqrt
    # etc.) allocates one large temporary buffer across *all* parameters' optimizer
    # state at once -- fixed overhead independent of batch_size, enough on its own to
    # OOM a 1.1B-param model on a 24GB card regardless of how small batch_size is.
    # foreach=False falls back to the traditional per-parameter loop -- slightly slower
    # per step, but avoids that large one-shot allocation.
    optimizer = torch.optim.AdamW(model.parameters(), lr=get_lr(start_it, max_iters),
                                   weight_decay=weight_decay, foreach=False)

    watch = {
        "Whead": model.head.weight,
        "layer0.qkv_proj (Wq/Wk/Wv)": model.blocks[0].attn.qkv_proj.weight,
    }

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    stale_evals = 0
    last_grad_norm = None
    last_train_loss = None
    last_val_loss = None
    last_lr = get_lr(start_it, max_iters)
    stopped_early = False

    def checkpoint_path():
        return os.path.join(out_dir, "kjeldchat.pt")

    def best_checkpoint_path():
        return os.path.join(out_dir, "kjeldchat_best.pt")

    def save_checkpoint(it, path=None):
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "iter": it,
                "val_loss": last_val_loss,
                "best_val_loss": best_val_loss,
                "best_iter": best_iter,
                "resumed_from": args.resume,
            },
            path or checkpoint_path(),
        )

    for it in range(start_it, max_iters + 1):
        now = time.time()
        elapsed = now - t0
        steps_this_run = it - start_it
        tok_per_sec = (steps_this_run * tokens_per_step) / elapsed if steps_this_run > 0 else 0.0

        due_for_eval = it == start_it or it % eval_every == 0 or it == max_iters
        due_for_checkpoint = it == start_it or it % checkpoint_every == 0 or it == max_iters
        due_for_console = it == start_it or it % console_every == 0

        if due_for_eval:
            losses = estimate_loss(model, data, val_windows)
            last_train_loss = losses["train"]
            last_val_loss = losses["val"]
            train_ppl = math.exp(min(losses["train"], 20))
            val_ppl = math.exp(min(losses["val"], 20))

            improved = losses["val"] < best_val_loss
            if improved:
                best_val_loss = losses["val"]
                best_iter = it
                stale_evals = 0
                status = "* new best *"
                # Saved immediately, separately from the periodic/final checkpoint --
                # the periodic cadence (checkpoint_every) can land on a step other than
                # the actual best one, so relying on it alone risks the true best-val
                # weights never landing on disk. This guarantees they always do.
                save_checkpoint(it, best_checkpoint_path())
            else:
                stale_evals += 1
                status = f"(no improvement: {stale_evals}/{args.patience})"

            print(
                f"[eval] step {it:5d}/{max_iters} | train loss {losses['train']:.4f} (ppl {train_ppl:6.1f}) | "
                f"val loss {losses['val']:.4f} (ppl {val_ppl:6.1f}) | best val {best_val_loss:.4f}@{best_iter} | "
                f"lr {last_lr:.2e} | grad_norm {f'{last_grad_norm:.2f}' if last_grad_norm else 'n/a'} | "
                f"{tok_per_sec:,.0f} tok/s | {elapsed:.0f}s elapsed | {status}"
            )

            if stale_evals >= args.patience:
                print(f"val loss hasn't improved for {args.patience} consecutive evals -- "
                      f"stopping early at step {it} (best: val loss {best_val_loss:.4f} at step {best_iter})")
                save_checkpoint(it)
                stopped_early = True
                break

        if due_for_checkpoint:
            save_checkpoint(it)
            print(f"[checkpoint] saved at step {it} (val_loss {last_val_loss:.4f}, "
                  f"best {best_val_loss:.4f}@{best_iter})")

        if due_for_console and not due_for_eval:
            train_loss_str = f"{last_train_loss:.4f}" if last_train_loss is not None else "n/a"
            print(
                f"[console] step {it:5d}/{max_iters} | train loss {train_loss_str} | "
                f"lr {last_lr:.2e} | grad_norm {f'{last_grad_norm:.2f}' if last_grad_norm else 'n/a'} | "
                f"{tok_per_sec:,.0f} tok/s | {elapsed:.0f}s elapsed"
            )

        if it == max_iters:
            break

        last_lr = get_lr(it, max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = last_lr

        x, y = get_batch("train", data)
        with autocast_ctx:
            _, loss = model(x, y)
        last_train_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

        debug_this_step = DEBUG and due_for_eval
        if debug_this_step:
            print(f"  [debug] this batch's loss: {loss.item():.4f}")
            print("  [debug] gradient norms by weight group:")
            print(f"    tok_emb + pos_emb        {grad_norm([model.tok_emb.weight, model.pos_emb.weight]):.4f}")
            for i, block in enumerate(model.blocks):
                qkv = grad_norm([block.attn.qkv_proj.weight])
                wo = grad_norm([block.attn.out_proj.weight])
                mlp = grad_norm([block.mlp.fc.weight, block.mlp.proj.weight])
                print(f"    layer {i}: qkv_proj(Wq/Wk/Wv)={qkv:.4f}  out_proj(Wo)={wo:.4f}  mlp(W1/W2)={mlp:.4f}")
            print(f"    Whead                    {grad_norm([model.head.weight]):.4f}")
            before = {name: w.detach().clone() for name, w in watch.items()}

        optimizer.step()

        if debug_this_step:
            print("  [debug] weight nudge this step:")
            for name, w in watch.items():
                change = (w.detach() - before[name]).norm().item()
                print(f"    {name}: ||change|| = {change:.6f}")
            print("")

    total_elapsed = time.time() - t0
    reason = "early stopping (val loss rising)" if stopped_early else "max_iters reached"
    print(
        f"done in {total_elapsed:.0f}s ({reason}). "
        f"best checkpoint: step {best_iter}, val loss {best_val_loss:.4f}, "
        f"saved to {best_checkpoint_path()} (latest step {it} saved to {checkpoint_path()})"
    )


if __name__ == "__main__":
    main()
