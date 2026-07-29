"""
Tokenizes finetune_corpus.txt for finetuning -- parallel to ../../base/data/tokenize_dataset.py,
but aware of each Q/A pair's prompt/answer boundary. Alongside the token ids, writes a
parallel mask array (1 = loss counted, 0 = masked prompt token) so finetune_train.py
can set masked positions' targets to -100 and train on producing the answer, not
reproducing the question (model.py's cross_entropy already ignores index -100).

Run shuffle_finetune.py first so the val split (the last val_fraction of pairs) is a random
sample rather than whichever pairs generate_qa.py happened to write last.

Run from within finetune/data/ (see ../FINETUNE_PARAMS.md):
    cd finetune/data
    python3 tokenize_finetune.py                # finetune_corpus_shuffled.txt -> train/val .bin + meta.json
"""

import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer

EOT = "<|endoftext|>"
ANSWER_MARKER = "\nAnswer:"

# Hard safety net for the model's block_size=1024: generate_qa.py caps passages at
# MAX_PASSAGE_TOKENS=400 tokens so a typical Context+Question+Answer example lands
# nowhere near this, but it's an estimate (char-based extraction, then token
# truncation) applied before the actual question/answer text exists -- an unusually
# long question or answer could still push a full example over budget. Rather than
# silently truncating (which could cut off part of the answer, breaking the model's
# grounding on the passage), any example over this length is dropped from training
# entirely.
MAX_EXAMPLE_TOKENS = 900


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
        prefix = block[:idx] + ANSWER_MARKER
        full = block + f"{EOT}\n"
        pairs.append((prefix, full))
    return pairs, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str,
                         default=os.path.join(os.path.dirname(__file__), "finetune_corpus_shuffled.txt"))
    # Same tokenizer as the base model, at the repo root's base/data/ -- must match for
    # finetuning to mean anything against the base checkpoint's embeddings.
    parser.add_argument("--tokenizer_path", type=str,
                         default=os.path.join(os.path.dirname(__file__), "..", "..",
                                               "base", "data", "tokenizer", "tokenizer.json"))
    parser.add_argument("--out_dir", type=str, default=os.path.dirname(__file__) or ".")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="fraction of pairs (from the end of the already-shuffled "
                              "file) held out for validation")
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"{args.tokenizer_path} not found -- run vocab_dataset.py first")

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id is None:
        raise ValueError(f"{EOT!r} not found in tokenizer vocab -- was it trained with vocab_dataset.py?")
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    pairs, skipped = load_pairs(args.in_path)
    print(f"loaded {len(pairs)} Q/A pairs from {args.in_path} ({skipped} skipped -- no "
          f"'{ANSWER_MARKER}' marker found)")

    ids_chunks = []
    mask_chunks = []
    boundary_mismatches = 0
    oversized = 0
    for prefix, full in pairs:
        full_ids = tokenizer.encode(full).ids
        if len(full_ids) > MAX_EXAMPLE_TOKENS:
            oversized += 1
            continue
        prefix_ids = tokenizer.encode(prefix).ids
        split = len(prefix_ids)
        # BPE merges are supposed to leave a stable boundary right at "Answer:" (a byte-
        # level tokenizer attaches leading spaces to the *following* word, so nothing
        # should merge across it) -- but if it ever doesn't, fall back to training that
        # one pair unmasked rather than silently mis-masking it.
        if full_ids[:split] != prefix_ids:
            boundary_mismatches += 1
            split = 0
        mask = [0] * split + [1] * (len(full_ids) - split)
        ids_chunks.append(np.array(full_ids, dtype=dtype))
        mask_chunks.append(np.array(mask, dtype=np.uint8))
    # Dropping oversized examples doesn't disturb the val split below: finetune_corpus.txt
    # was already pair-shuffled by shuffle_finetune.py, so whichever pairs survive are still
    # in random order relative to each other -- "last val_fraction of what's kept" is
    # still a random sample, not biased toward whatever happened to follow a drop.
    kept_pairs = len(ids_chunks)

    ids = np.concatenate(ids_chunks)
    mask = np.concatenate(mask_chunks)
    print(f"encoded {len(ids):,} tokens from {kept_pairs}/{len(pairs)} pairs "
          f"({oversized} dropped for exceeding {MAX_EXAMPLE_TOKENS} tokens, "
          f"{boundary_mismatches} had a prompt/answer boundary mismatch and were left unmasked)")

    n_val = int(kept_pairs * args.val_fraction)
    n_train_pairs = kept_pairs - n_val
    train_token_count = sum(len(c) for c in ids_chunks[:n_train_pairs])

    os.makedirs(args.out_dir, exist_ok=True)
    ids[:train_token_count].tofile(os.path.join(args.out_dir, "train_ids.bin"))
    mask[:train_token_count].tofile(os.path.join(args.out_dir, "train_mask.bin"))
    ids[train_token_count:].tofile(os.path.join(args.out_dir, "val_ids.bin"))
    mask[train_token_count:].tofile(os.path.join(args.out_dir, "val_mask.bin"))

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "eot_id": eot_id,
        "eot_token": EOT,
        "tokenizer_path": args.tokenizer_path,
        "train_tokens": int(train_token_count),
        "val_tokens": int(len(ids) - train_token_count),
        "num_pairs": kept_pairs,
        "num_val_pairs": n_val,
        "oversized_dropped": oversized,
        "boundary_mismatches": boundary_mismatches,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out_dir}/{{train,val}}_{{ids,mask}}.bin and meta.json "
          f"({meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens)")


if __name__ == "__main__":
    main()
