"""
Exports ../../finetune/data/finetune_corpus_shuffled.txt -- the exact same Q/A corpus
KjeldChat was finetuned on -- into prompt/completion JSONL for finetuning Llama with
transformers + trl's SFTTrainer, instead of tokenize_finetune.py's pre-tokenized .bin
format (which is specific to this project's own byte-level tokenizer and model.py's
GPTConfig, neither of which Llama uses).

Reproduces tokenize_finetune.py's train/val split exactly -- same corpus, same boundary
logic (last --val_fraction of pairs, snapped forward to the next Context change so no
passage straddles the split, matching v7's "honest" context-grouped validation) -- so
the Llama run's val set is the same kind of held-out measurement as KjeldChat's, not a
different, incomparable one.

Deliberately does NOT pre-filter by exact Llama token count here (that needs the gated
tokenizer, which may not be available until the pod's HF token/license access is set
up) -- SFTTrainer's own max_seq_length truncates/filters oversized examples at train
time instead.

Writes {"prompt": "Context: ...\\nQuestion: ...\\nAnswer:", "completion": " ...<eos>"} --
trl's SFTTrainer masks the prompt half automatically for this exact two-column format,
no manual DataCollatorForCompletionOnlyLM needed.

Run from within llama/data/:
    python3 export_sft_data.py
"""
import argparse
import json
import os

EOT = "<|endoftext|>"
ANSWER_MARKER = "\nAnswer:"
NO_CONTEXT_LINE = "Context: N/A"


def group_key(prefix):
    """Must match finetune/data/shuffle_finetune.py's group_key exactly -- see
    tokenize_finetune.py's identical helper for why."""
    context_line = prefix.split("\n", 1)[0]
    return prefix if context_line == NO_CONTEXT_LINE else context_line


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    pairs = []
    skipped = 0
    for block in text.split(f"{EOT}\n"):
        if not block.strip():
            continue
        idx = block.find(ANSWER_MARKER)
        if idx == -1:
            skipped += 1
            continue
        prompt = block[:idx] + ANSWER_MARKER
        completion = block[idx + len(ANSWER_MARKER):]
        pairs.append((prompt, completion))
    return pairs, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str,
                         default=os.path.join(os.path.dirname(__file__), "..", "..",
                                               "finetune", "data", "finetune_corpus_shuffled.txt"))
    parser.add_argument("--out_dir", type=str, default=os.path.dirname(__file__) or ".")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--eos_token", type=str, default="<|end_of_text|>",
                         help="Llama 3.2's EOS token (appended to every completion so the "
                              "model learns to stop) -- override if using a tokenizer "
                              "with a different one")
    args = parser.parse_args()

    pairs, skipped = load_pairs(args.in_path)
    print(f"loaded {len(pairs)} Q/A pairs from {args.in_path} ({skipped} skipped -- no "
          f"'{ANSWER_MARKER}' marker found)")

    n_val = int(len(pairs) * args.val_fraction)
    n_train = len(pairs) - n_val
    contexts = [group_key(prompt) for prompt, _ in pairs]
    moved = 0
    while n_train < len(pairs) and contexts[n_train - 1] == contexts[n_train]:
        n_train += 1
        moved += 1
    n_val = len(pairs) - n_train
    print(f"snapped the val boundary {moved} pair(s) forward to a Context change -- "
          f"no Context has pairs in both train and val")

    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, split_pairs in (("train", pairs[:n_train]), ("val", pairs[n_train:])):
        out_path = os.path.join(args.out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for prompt, completion in split_pairs:
                f.write(json.dumps({"prompt": prompt, "completion": completion + args.eos_token}) + "\n")
        print(f"wrote {len(split_pairs)} pairs to {out_path}")

    meta = {"num_pairs": len(pairs), "num_train_pairs": n_train, "num_val_pairs": n_val,
            "source": args.in_path}
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
