# KjeldGPT 1.1B — combined corpus training run

**Completed.** The run finished its full 3-epoch budget at step 808,998, final val loss
2.5010. The result is `checkpoints/kjeldgpt.pt` -- the frozen base every finetuning
round resumes from.

**This is still the base model's corpus -- unchanged.** Downstream, `finetune/`'s Q/A
finetuning corpus and RAG passage index moved to Wikipedia-only (see
`FINETUNE_PARAMS.md`'s "Wikipedia-only" section for why), but that's a decision about
what the *finetuned* model is grounded on at inference time, not about what this base
model is pretrained on -- KjeldGPT 1.1B's own knowledge still comes from the combined
Gutenberg + Wikipedia corpus described below.

## Corpus

| Source | Size | Files |
|---|---|---|
| Gutenberg (`data/gutenberg/clean_gutenberg`) | 12.0 GB | 30,817 books |
| Wikipedia (`data/wikipedia/clean_wikipedia`) | 19.48 GB | 6,083,989 articles / 2,028 shards |
| **Combined** | **~31.5 GB** | **32,845 files** |

Exact tokens (from `meta.json`): **7,363,682,333 total** — 6,627,314,100 train /
736,368,233 val.

## Tokenizer

- `vocab_size` = 50,257 (GPT-2-sized; bumped from 40,000 to cover Wikipedia's broader
  vocabulary on top of Gutenberg)
- Trained jointly on the combined corpus via `data/vocab_dataset.py`
- Saved to `data/tokenizer/tokenizer.json`

## Model architecture

| Param | Value |
|---|---|
| `n_layer` | 36 |
| `n_head` | 24 |
| `n_embd` | 1536 |
| `block_size` | 1024 |
| `vocab_size` | 50,257 |
| `tied` | True |
| `dropout` | 0.05 |
| **Total params** | **~1,098.7M (~1.1B)** |

Sized from the corpus itself (not a fixed time budget): ~3 epochs over the combined
corpus lands near the ~20-tokens/param compute-optimal ratio.

## Batch / memory

- `batch_size` = 24
- Measured peak VRAM: 79.5GB (RTX PRO 6000 Blackwell, 97.9GB total, ~18GB headroom)
- ~30,700 tok/s at this batch size

## Optimizer / LR schedule

| Param | Value |
|---|---|
| `peak_lr` | 2e-4 (down from the previous 294M run's 3e-4 — interpolated from the GPT-3 paper's params-to-LR table for a ~1.1B model) |
| `min_lr` | 2e-5 (peak/10) |
| `warmup_iters` | 1000 |
| `weight_decay` | 0.1 |
| Schedule | warmup + single cosine decay over `max_iters` |

## Training loop

| Param | Value |
|---|---|
| `max_iters` | 808,998 (3 epochs × 6,627,314,100 real train tokens ÷ (24×1024)) |
| `console_interval_seconds` | 300 (status line only -- last known train loss, no eval) |
| `eval_interval_seconds` | 1800 (30 min -- infrequent by design, eval is pure overhead) |
| `eval_iters` | 150 (batches averaged per eval, per split -- bumped from 50 after a false-positive early stop at step 21,221) |
| `checkpoint_interval_seconds` | 3600 (1 hour -- saves *current* weights on a fixed clock, decoupled from whether the last eval improved) |
| `patience` | 200 (passed as `--patience 200`; the code default is 3). Early-stopping evals -- a safety net against genuine multi-hour divergence, not a routine trigger; the run is sized via `max_iters` for exactly 3 epochs, which is the intended stopping point |
| `out_dir` | `checkpoints` |

`base_train.py` writes a single `ckpt.pt` into `out_dir`, overwritten on each save. The
finished run's copy is what lives locally as `base/checkpoints/kjeldgpt.pt`.

Console/eval/checkpoint are independently timed so each can be tuned for what it's
actually for: console is free (reuses the last-computed train loss, no extra forward
passes), eval is genuine overhead (150 x2-split forward passes, ~87s) so kept
infrequent, and checkpointing is a multi-GB disk write decoupled from "did val loss
improve" -- it saves the latest weights every hour regardless, bounding crash/restart
loss to at most an hour rather than "however long since the last lucky improving eval."

## Precision & gradient handling

- bf16 autocast on the forward pass (weights/optimizer state stay fp32)
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)`
- Gradient accumulation: not used (batch_size=24 already fits comfortably; throughput
  was already flattening out by batch 24-28)

## Hardware

This run used a single RTX PRO 6000 Blackwell (97.9GB VRAM, 32 vCPU, 188GB RAM) rented
by the hour. Nothing in the code assumes that specific machine or provider -- any single
CUDA GPU works, and `batch_size` is the knob to turn for a smaller one. The batch size
here was chosen for throughput, not because the model needs 80GB: measured peak VRAM was
79.5GB at `batch_size=24`, and it scales down roughly linearly.

## Training time

~7.5 days at ~30,700 tok/s (3 epochs × 6,627,314,100 real train tokens).

## Launch command

Run `data/tokenize_dataset.py` first, then from within `base/`:

```
python base_train.py
```

It is a multi-day run, so start it under `tmux`/`screen`/`nohup` if the session can
drop. Training resumes from the last hourly checkpoint with
`--resume checkpoints/ckpt.pt` -- the LR schedule is a pure function of absolute step,
so a resumed run continues the same warmup/cosine curve rather than restarting it.
