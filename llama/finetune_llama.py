"""
Finetunes meta-llama/Llama-3.2-1B (base, NOT -Instruct -- see the project's reasoning:
an already chat-tuned model would confound "does more pretraining help" with "does the
existing chat template fight the Context/Question/Answer format") on the same corpus
and the same recipe shape as ../../finetune/finetune_train.py used for KjeldGPT's own
from-scratch model:

  - same corpus (data/train.jsonl / val.jsonl, exported by data/export_sft_data.py from
    the identical finetune_corpus_shuffled.txt KjeldChat trained on)
  - same masked objective: only completion tokens count toward the loss, never the
    Context/Question prompt -- trl's SFTTrainer does this automatically for a
    prompt/completion-format dataset, no manual DataCollatorForCompletionOnlyLM needed
  - same peak_lr/min_lr/cosine-schedule shape as KjeldChat's Stage 3b recipe (1e-5 peak,
    peak/10 min, warmup, cosine decay) -- see finetune/FINETUNE_PARAMS.md's "Batch /
    schedule" section for why this LR (an order of magnitude below typical pretraining
    LR, chosen so a narrow finetuning corpus adapts the model instead of overwriting
    what its (much larger, here) pretraining taught it)
  - same 3-epoch budget, same eval-driven early stopping via patience

Full finetuning (not LoRA), matching KjeldChat's own recipe -- the point of this run is
"same recipe, more pretraining," and LoRA would introduce a second confound (adapter
capacity) on top of that.

Run on the pod (needs a GPU, ANTHROPIC not required here; needs a HuggingFace token with
meta-llama/Llama-3.2-1B access accepted -- huggingface-cli login first):
    pip install transformers trl accelerate
    cd llama
    python3 finetune_llama.py
"""
import argparse
import json
import os
import shutil

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer


class PermanentSnapshotCallback(TrainerCallback):
    """Copies checkpoint-<step> to a permanently-kept snapshots/step<N> directory every
    --snapshot_every steps, bypassing save_total_limit's rolling-window pruning --
    mirrors KjeldChat's own Round 5 methodology (finetune/FINETUNE_PARAMS.md's
    "Checkpoint selection"): the lowest-val-loss checkpoint isn't always the
    best-testing one (v7 itself: best-by-test was step 2700, best-by-val-loss was step
    3450), so keeping periodic snapshots lets a later blackbox sweep pick empirically
    instead of trusting eval_loss alone."""

    def __init__(self, snapshot_every, out_dir):
        self.snapshot_every = snapshot_every
        self.out_dir = out_dir

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.snapshot_every != 0:
            return
        src = os.path.join(self.out_dir, f"checkpoint-{step}")
        dst = os.path.join(self.out_dir, "snapshots", f"step{step}")
        if os.path.isdir(src) and not os.path.isdir(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
            print(f"[snapshot] permanently saved step {step} to {dst}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--data_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
    # Same recipe shape as finetune/finetune_train.py's Stage 3b (see module docstring).
    parser.add_argument("--peak_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4,
                         help="per-device batch size -- default 4 (with --grad_accum 2, "
                              "effective batch size 8, same as KjeldChat's recipe) rather "
                              "than 8 directly, sized for a 32GB-class GPU doing full "
                              "finetuning; raise toward 8 on a larger GPU")
    parser.add_argument("--grad_accum", type=int, default=2,
                         help="effective batch size is batch_size * grad_accum -- kept "
                              "at 8 to match KjeldChat's recipe regardless of how the "
                              "two factors split")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                         default=True, help="trades compute for activation memory -- on "
                              "by default since full finetuning a 1.24B model at "
                              "batch_size=4/seq_len=1024 is tight on a 32GB GPU otherwise")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                         help="matches KjeldChat's block_size=1024 -- keeps the recipe "
                              "comparable rather than exploiting Llama's much longer "
                              "native context window")
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10,
                         help="early-stopping patience in eval_steps units, same as "
                              "KjeldChat's Round 4 recipe (--patience 10)")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--snapshot_every", type=int, default=460,
                         help="permanently keep a checkpoint every N steps (see "
                              "PermanentSnapshotCallback), independent of save_total_limit's "
                              "rolling window. NOT KjeldChat's Round 5 --snapshot_every 100 "
                              "directly -- SFTTrainer's steps are pair-batched, not "
                              "token-windowed like finetune_train.py's (see the module's "
                              "step-count note), so 100 of KjeldChat's steps out of its "
                              "3,789-step run is proportionally ~460 of this run's 17,277 "
                              "steps -- same ~2.6%%-of-run granularity (~38 snapshots "
                              "either way), not a literal '100' that'd yield ~175 here")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                         help="path to a checkpoint-<N> directory to resume from "
                              "(optimizer/scheduler state included) rather than starting fresh")
    args = parser.parse_args()

    print(f"loading {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    train_ds = load_dataset("json", data_files=os.path.join(args.data_dir, "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=os.path.join(args.data_dir, "val.jsonl"), split="train")
    print(f"train: {len(train_ds)} pairs, val: {len(val_ds)} pairs", flush=True)

    min_lr_ratio = 0.1  # min_lr = peak_lr / 10, same as KjeldChat's recipe

    config = SFTConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.peak_lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": min_lr_ratio},
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_length=args.max_seq_length,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to=[],
        logging_steps=5,
        completion_only_loss=True,  # only completion tokens count toward the loss --
                                     # the prompt/completion dataset format's whole point
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience),
                   PermanentSnapshotCallback(args.snapshot_every, args.out_dir)],
    )

    print("starting training ...", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    best_path = os.path.join(args.out_dir, "best")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"saved best checkpoint to {best_path}", flush=True)


if __name__ == "__main__":
    main()
