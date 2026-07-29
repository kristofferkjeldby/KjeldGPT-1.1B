"""
Training loop for the GPT in model.py -- a from-scratch run sized for the combined
Gutenberg + Wikipedia corpus prepared by data/clean_gutenberg_dataset.py,
data/vocab_dataset.py, and data/tokenize_dataset.py (see BASE_PARAMS.md). Run from
within base/ (data/, checkpoints/ are its siblings here).

Run data/tokenize_dataset.py once first. Then:
    python base_train.py
    python base_train.py --debug            # also print gradient norms + weight-nudge
                                                 # magnitude at every eval
    python base_train.py --patience 5        # allow more consecutive non-improving
                                                 # evals before stopping early (default 3)
    python base_train.py --resume checkpoints/ckpt.pt
                                                 # continue an existing checkpoint -- the lr
                                                 # schedule is a pure function of absolute
                                                 # step, so resuming naturally continues
                                                 # wherever the warmup/cosine curve was

Training stops automatically once val loss has risen (failed to improve) for
`--patience` consecutive evals -- so it's safe to leave max_iters high and let this cut
training short instead of babysitting the loss curve.

Uses a single warmup + cosine decay schedule computed for the full max_iters budget
upfront -- smoother annealing than discrete step-decay, and this is a from-scratch run
so there's no existing schedule state to disrupt.
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

# model.py lives at the repo root; this script lives in base/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig

# Plain print() doesn't reliably flush on every newline through the tmux ->
# base_train.py process chain (depends on Python's tty detection). Force
# line-buffering explicitly so progress prints show up immediately instead of sitting
# in a buffer until it fills or the process exits.
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true",
                     help="print gradient norms and weight-nudge magnitudes at every eval")
parser.add_argument("--patience", type=int, default=3,
                     help="stop training after this many consecutive evals with no "
                          "val loss improvement (early stopping against overfitting)")
parser.add_argument("--resume", type=str, default=None,
                     help="path to a checkpoint to resume from -- continues the step "
                          "counter, lr schedule, and best-val tracking")
parser.add_argument("--peak_lr", type=float, default=None,
                     help="override the peak learning rate below")
parser.add_argument("--out_dir", type=str, default=None,
                     help="override the checkpoint save directory below -- note --resume "
                          "only controls where to *load* from, checkpoints are always "
                          "*saved* to this dir regardless, so use this to test/smoke-run "
                          "without touching your real checkpoint")
args = parser.parse_args()
DEBUG = args.debug

# ---- hyperparameters -------------------------------------------------------
# Sized for the combined ~31.5GB / ~7.8B-token corpus (Gutenberg + full English
# Wikipedia) on an RTX PRO 6000 Blackwell (98GB VRAM). ~1.1B params targets ~3 epochs
# over this corpus at the ~20-tokens/param compute-optimal ratio -- chosen from the
# corpus size itself rather than a fixed time budget -- the real constraint is how
# much unique data exists, not how long the hardware is available.
batch_size = 24          # measured directly on this GPU/architecture: 79.5GB peak at
                         # batch=24 (~18GB margin), vs. 84.9GB at 26 and OOM at 32 --
                         # tok/s was already flattening out by 24-28, so no real
                         # throughput to trade away for the extra headroom
block_size = 1024
n_layer = 36
n_head = 24
n_embd = 1536
dropout = 0.05          # modest hedge against overfitting -- only ~3 epochs over the
                         # corpus, so overfitting risk is low to begin with
tied = True              # clean from-scratch run -- standard GPT-2 practice, no old
                         # checkpoint to perturb by introducing this after the fact

max_iters = 808998       # 3 epochs x 6,627,314,100 real train tokens (from meta.json)
                         # / (batch_size * block_size)
warmup_iters = 1000
peak_lr = args.peak_lr if args.peak_lr is not None else 2e-4
                         # interpolated from the GPT-3 paper's params-to-LR table for a
                         # ~1.1B model (between 760M's 2.5e-4 and 1.3B's 2e-4), since
                         # larger models generally need a lower peak LR for stable
                         # training
min_lr = peak_lr / 10
weight_decay = 0.1
# Three independent cadences, decoupled so each can be tuned for what it's actually for
# instead of one shared interval forcing a compromise between them -- this run is meant
# to go unattended for long stretches:
#   - console: cheap, frequent feedback (just the last-computed train loss -- already
#     free, no extra forward passes) so there's still visible signal between evals
#   - eval: the expensive part (eval_iters forward passes x2 splits) -- infrequent, since
#     its overhead is pure throughput loss; also drives best-val tracking and the
#     patience/early-stop safety net, so still needs to happen regularly enough to mean
#     something
#   - checkpoint: a periodic disk write (a multi-GB file to network storage), decoupled
#     from "did val loss improve" -- saves *current* weights on a fixed cadence instead,
#     so a crash loses at most one interval's worth of progress, not "however long since
#     the last lucky improving eval"
console_interval_seconds = 300      # 5 min -- just a status line, no eval triggered
eval_interval_seconds = 1800        # 30 min -- eval_iters=150 x2 splits costs ~87s; at
                                     # this cadence that's ~5% overhead instead of the
                                     # ~29%/~14% it was at 300s/600s
checkpoint_interval_seconds = 3600  # 1 hour -- matches the "lose at most an hour if it
                                     # stalls or crashes" tolerance for running unattended
eval_iters = 150         # batches averaged when estimating loss -- val loss drives
                         # early stopping, so a noisy estimate risks a false "no
                         # improvement" streak triggering it well before any real
                         # overfitting-driven degradation could plausibly start on a
                         # non-repeating corpus this size. More batches means a less
                         # noisy estimate, without eval cost meaningfully eating into
                         # the eval_interval_seconds budget

out_dir = args.out_dir if args.out_dir is not None else "checkpoints"
data_dir = "data"
# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# bf16 autocast: casts matmuls/activations to bf16 within the context while keeping
# weights and optimizer state in fp32 -- roughly halves activation memory and speeds up
# matmuls on Ampere+ tensor cores (this GPU generation supports bf16 natively). No
# GradScaler needed, unlike fp16 -- bf16 has the same exponent range as fp32, so it
# doesn't underflow the way fp16 does, just less mantissa precision.
autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" \
    else contextlib.nullcontext()


def get_lr(it):
    if it < warmup_iters:
        return peak_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0 over the decay window
    return min_lr + coeff * (peak_lr - min_lr)


def load_data():
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.dtype(meta["dtype"])
    train_ids = np.memmap(os.path.join(data_dir, "train.bin"), dtype=dtype, mode="r")
    val_ids = np.memmap(os.path.join(data_dir, "val.bin"), dtype=dtype, mode="r")
    return train_ids, val_ids, meta["vocab_size"]


def get_batch(split, train_ids, val_ids):
    data = train_ids if split == "train" else val_ids
    # Pick batch_size random starting points, each giving a block_size chunk as input
    # and the same chunk shifted by one token as the target (next-token prediction).
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def grad_norm(params):
    """L2 norm of gradients across a group of parameters (None if none have grads yet)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None
    return torch.sqrt(sum((g ** 2).sum() for g in grads)).item()


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids):
    # Average loss over several batches per split -- a single batch is too noisy to
    # compare training progress reliably.
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split, train_ids, val_ids)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"device: {device}  patience: {args.patience} evals")
    train_ids, val_ids, vocab_size = load_data()
    tokens_per_step = batch_size * block_size
    print(f"data: {len(train_ids):,} train tokens, {len(val_ids):,} val tokens, vocab_size={vocab_size}")
    print(f"hyperparams: n_layer={n_layer} n_head={n_head} n_embd={n_embd} block_size={block_size} "
          f"batch_size={batch_size} tied={tied} weight_decay={weight_decay} dropout={dropout}")
    print(f"lr schedule: warmup_iters={warmup_iters} peak_lr={peak_lr:.2e} min_lr={min_lr:.2e} "
          f"max_iters={max_iters}")
    print(f"console every ~{console_interval_seconds}s, eval every ~{eval_interval_seconds}s, "
          f"checkpoint every ~{checkpoint_interval_seconds}s ({tokens_per_step:,} tokens/step)")

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
    # True only if this build of model.py ties tok_emb/head -- keeping this check (rather
    # than hardcoding) means the resume logic below stays correct regardless of the tied
    # setting above.
    model_ties_weights = model.head.weight is model.tok_emb.weight

    start_it = 0
    best_val_loss = float("inf")
    best_iter = 0
    if args.resume:
        torch.serialization.add_safe_globals([GPTConfig])
        ckpt = torch.load(args.resume, map_location=device)
        raw_sd = ckpt["model"]
        # Relevant only when the *current* model ties weights but the checkpoint being
        # resumed doesn't (tok_emb.weight and head.weight are genuinely different tensors
        # there). state_dict() always includes both key names regardless of tying (PyTorch
        # doesn't dedupe aliased parameters by name), so checking value equality -- not
        # just key presence -- is what actually distinguishes "this checkpoint predates
        # tying" from "this checkpoint already ties, and the duplicate key is cosmetic".
        # load_state_dict won't flag a genuinely different head.weight as "unexpected"
        # either way (self.head is still a real module) -- it'll silently let whichever of
        # tok_emb.weight/head.weight loads last overwrite the other, discarding one of two
        # independently-trained matrices outright, a much bigger perturbation than tying
        # is supposed to cause. Average them instead as the tied weight's starting point.
        architecture_changed = (model_ties_weights
                                 and "head.weight" in raw_sd
                                 and not torch.equal(raw_sd["tok_emb.weight"], raw_sd["head.weight"]))
        if architecture_changed:
            averaged = (raw_sd["tok_emb.weight"] + raw_sd["head.weight"]) / 2
            raw_sd = dict(raw_sd)
            raw_sd["tok_emb.weight"] = averaged
            del raw_sd["head.weight"]
            print("resume: checkpoint predates weight tying -- averaging the old "
                  "tok_emb.weight/head.weight matrices as the tied weight's starting "
                  "point instead of discarding one of them")
        model.load_state_dict(raw_sd, strict=False)
        start_it = ckpt["iter"] + 1
        if architecture_changed:
            print("  resetting best-val tracking (patience budget) since the old value "
                  "isn't a fair comparison against the new architecture")
            best_val_loss = float("inf")
            best_iter = start_it
        else:
            # best_val_loss/best_iter are separate fields since a saved checkpoint's
            # weights are just "current", not necessarily the best-ever -- fall back to
            # the old single val_loss field for checkpoints that predate this, where it
            # *was* the best-at-save-time value.
            best_val_loss = ckpt.get("best_val_loss", ckpt["val_loss"])
            best_iter = ckpt.get("best_iter", ckpt["iter"])
        print(f"resumed from {args.resume}: step {ckpt['iter']}, val_loss {ckpt['val_loss']:.4f}, "
              f"best_val_loss {best_val_loss:.4f}@{best_iter}")
        # optimizer state (momentum/variance) isn't checkpointed, so AdamW restarts fresh
        # here -- a brief re-warmup of its adaptive estimates, not a correctness issue

    optimizer = torch.optim.AdamW(model.parameters(), lr=get_lr(start_it), weight_decay=weight_decay)

    # Two weights to watch get "nudged" when --debug is on: one at the very end of the
    # stack (Whead), one buried in the first layer -- shows the update reaches all the
    # way back through all n_layer layers, not just the last one.
    watch = {
        "Whead": model.head.weight,
        "layer0.qkv_proj (Wq/Wk/Wv)": model.blocks[0].attn.qkv_proj.weight,
    }

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    last_console_time = t0
    last_eval_time = t0
    last_checkpoint_time = t0

    stale_evals = 0  # consecutive evals without a val loss improvement
    last_grad_norm = None
    last_train_loss = None
    last_val_loss = ckpt["val_loss"] if args.resume else None
    last_lr = get_lr(start_it)
    stopped_early = False

    def checkpoint_path():
        return os.path.join(out_dir, "ckpt.pt")

    def save_checkpoint(it):
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "iter": it,
                "val_loss": last_val_loss,
                "best_val_loss": best_val_loss,
                "best_iter": best_iter,
            },
            checkpoint_path(),
        )

    for it in range(start_it, max_iters + 1):
        now = time.time()
        elapsed = now - t0
        steps_this_run = it - start_it
        tok_per_sec = (steps_this_run * tokens_per_step) / elapsed if steps_this_run > 0 else 0.0

        due_for_eval = it == start_it or (now - last_eval_time) >= eval_interval_seconds
        due_for_checkpoint = it == start_it or (now - last_checkpoint_time) >= checkpoint_interval_seconds
        due_for_console = it == start_it or (now - last_console_time) >= console_interval_seconds

        if due_for_eval:
            last_eval_time = now
            losses = estimate_loss(model, train_ids, val_ids)
            last_train_loss = losses["train"]
            last_val_loss = losses["val"]
            train_ppl = math.exp(min(losses["train"], 20))  # cap to avoid inf on a bad early step
            val_ppl = math.exp(min(losses["val"], 20))

            improved = losses["val"] < best_val_loss
            if improved:
                best_val_loss = losses["val"]
                best_iter = it
                stale_evals = 0
                status = "* new best *"
            else:
                stale_evals += 1
                status = f"(no improvement: {stale_evals}/{args.patience})"

            print(
                f"[eval] step {it:6d}/{max_iters} | train loss {losses['train']:.4f} (ppl {train_ppl:6.1f}) | "
                f"val loss {losses['val']:.4f} (ppl {val_ppl:6.1f}) | best val {best_val_loss:.4f}@{best_iter} | "
                f"lr {last_lr:.2e} | grad_norm {f'{last_grad_norm:.2f}' if last_grad_norm else 'n/a'} | "
                f"{tok_per_sec:,.0f} tok/s | {elapsed:.0f}s elapsed | {status}"
            )

            if stale_evals >= args.patience:
                print(
                    f"val loss hasn't improved for {args.patience} consecutive evals -- stopping early "
                    f"at step {it} (best: val loss {best_val_loss:.4f} at step {best_iter})"
                )
                save_checkpoint(it)
                stopped_early = True
                break

        if due_for_checkpoint:
            last_checkpoint_time = now
            save_checkpoint(it)
            print(f"[checkpoint] saved at step {it} (val_loss {last_val_loss:.4f}, "
                  f"best {best_val_loss:.4f}@{best_iter})")

        if due_for_console and not due_for_eval:
            # Skip if an eval just printed -- that line already covers this step, no
            # need for a redundant console-only line right after it.
            last_console_time = now
            train_loss_str = f"{last_train_loss:.4f}" if last_train_loss is not None else "n/a"
            print(
                f"[console] step {it:6d}/{max_iters} | train loss {train_loss_str} | "
                f"lr {last_lr:.2e} | grad_norm {f'{last_grad_norm:.2f}' if last_grad_norm else 'n/a'} | "
                f"{tok_per_sec:,.0f} tok/s | {elapsed:.0f}s elapsed"
            )
        elif due_for_console:
            last_console_time = now

        last_lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = last_lr

        x, y = get_batch("train", train_ids, val_ids)
        with autocast_ctx:
            _, loss = model(x, y)
        last_train_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clips to norm 1.0 (standard across GPT pretraining recipes) -- cheap insurance
        # against a single bad batch spiking the gradient and derailing a multi-day run.
        # Returns the pre-clip total norm, reused directly instead of computing it twice.
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

    if not stopped_early:
        # Loop completed all max_iters rather than breaking early -- the periodic
        # checkpoint cadence may not have landed exactly on the last step, so save
        # explicitly here too rather than potentially losing however much ran since the
        # last periodic save.
        save_checkpoint(it)

    total_elapsed = time.time() - t0
    reason = "early stopping (val loss rising)" if stopped_early else "max_iters reached"
    print(
        f"done in {total_elapsed:.0f}s ({reason}). "
        f"best checkpoint: step {best_iter}, val loss {best_val_loss:.4f}, "
        f"saved to {os.path.join(out_dir, 'ckpt.pt')}"
    )


if __name__ == "__main__":
    main()
