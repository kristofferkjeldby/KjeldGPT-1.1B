"""
Encodes the combined corpus -- Gutenberg (clean_gutenberg/) plus Wikipedia
(clean_wikipedia/) -- into token ids using the tokenizer trained by vocab_dataset.py, and
writes train.bin + val.bin + meta.json for base_train.py to memmap directly. Each
cleaned file (a book, or a shard of Wikipedia articles already delimited by
<|endoftext|>) gets one more <|endoftext|> appended after it, so the model learns
document boundaries instead of reading everything as one continuous, unrelated stream.

This overwrites train.bin, val.bin, and meta.json in place if they already exist.

Run from within data/ (see ../BASE_PARAMS.md):
    cd data
    python tokenize_dataset.py                  # encode the combined corpus
    python tokenize_dataset.py --limit 200       # smoke test on a subset first

Requires: pip install tokenizers numpy tqdm
"""

import argparse
import json
import os
import time

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

EOT_TOKEN = "<|endoftext|>"

BATCH_SIZE = 32   # files per tokenizer.encode_batch() call -- encode_batch is Rust-side
                   # parallel (releases the GIL) across up to one thread per text in the
                   # batch, so this also caps how many texts are held in memory and
                   # processed concurrently at once. Much higher values are prone to
                   # hanging (unresponsive to SIGINT, genuinely stuck in the native call)
                   # and ballooning RSS on many-core boxes -- keep this conservative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, action="append", default=None,
                         help="dir of cleaned .txt files to encode -- pass multiple "
                              "times to override the default (Gutenberg + Wikipedia)")
    parser.add_argument("--tokenizer_path", type=str,
                         default=os.path.join("tokenizer", "tokenizer.json"))
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--limit", type=int, default=None,
                         help="only encode the first N cleaned files (smoke test)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="fraction of tokens (from the end of the concatenated "
                              "stream) held out for validation")
    args = parser.parse_args()
    clean_dirs = args.clean_dir or [
        os.path.join("gutenberg", "clean_gutenberg"),
        os.path.join("wikipedia", "clean_wikipedia"),
    ]
    tokenizer_path = args.tokenizer_path
    out_dir = args.out_dir

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"{tokenizer_path} not found -- run vocab_dataset.py first")
    for clean_dir in clean_dirs:
        if not os.path.isdir(clean_dir):
            raise FileNotFoundError(f"{clean_dir} not found -- run gutenberg/clean_gutenberg_dataset.py "
                                     f"or wikipedia/clean_wikipedia_dataset.py first")

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    if eot_id is None:
        raise ValueError(f"{EOT_TOKEN!r} not found in tokenizer vocab -- was it trained with vocab_dataset.py?")
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    log(f"loaded tokenizer: vocab_size={vocab_size}, eot_id={eot_id}, dtype={dtype.__name__}")

    paths = []
    for clean_dir in clean_dirs:
        filenames = sorted(os.listdir(clean_dir))
        paths.extend(os.path.join(clean_dir, f) for f in filenames)
    if args.limit:
        paths = paths[: args.limit]
    log(f"encoding {len(paths):,} files from {', '.join(clean_dirs)}")

    # Stream straight to a scratch file as each batch is encoded, rather than
    # accumulating every file's token array in RAM until the end -- at this corpus's
    # scale (~33k files, ~32GB text), holding all encoded tokens in memory at once risks
    # OOM. Peak memory here is bounded by one batch.
    os.makedirs(out_dir, exist_ok=True)
    eot_arr = np.array([eot_id], dtype=dtype)
    scratch_path = os.path.join(out_dir, "_all_tokens.bin")
    total_tokens = 0
    with open(scratch_path, "wb") as scratch:
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="encoding", unit="batch"):
            batch_paths = paths[i:i + BATCH_SIZE]
            texts = []
            for path in batch_paths:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            encodings = tokenizer.encode_batch(texts)
            for enc in encodings:
                ids = np.array(enc.ids, dtype=dtype)
                ids.tofile(scratch)
                eot_arr.tofile(scratch)
                total_tokens += len(ids) + 1

    log(f"encoded {total_tokens:,} tokens across {len(paths):,} files")

    n_val = int(total_tokens * args.val_fraction)
    n_train = total_tokens - n_val
    itemsize = np.dtype(dtype).itemsize
    train_bytes = n_train * itemsize

    # Split the scratch file into train.bin (first n_train tokens) and val.bin (the
    # rest) by copying in fixed-size chunks -- same low-memory reasoning as above, no
    # need to hold either file's full contents in RAM to do this split.
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    COPY_CHUNK = 100_000_000  # bytes per copy chunk
    bytes_seen = 0
    with open(scratch_path, "rb") as src, \
            open(train_path, "wb") as train_f, \
            open(val_path, "wb") as val_f:
        while True:
            chunk = src.read(COPY_CHUNK)
            if not chunk:
                break
            if bytes_seen < train_bytes:
                remaining_train = train_bytes - bytes_seen
                if len(chunk) <= remaining_train:
                    train_f.write(chunk)
                else:
                    train_f.write(chunk[:remaining_train])
                    val_f.write(chunk[remaining_train:])
            else:
                val_f.write(chunk)
            bytes_seen += len(chunk)
    os.remove(scratch_path)
    log(f"wrote {train_path} ({n_train:,} tokens) and {val_path} ({n_val:,} tokens)")

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "eot_id": eot_id,
        "eot_token": EOT_TOKEN,
        "tokenizer_path": tokenizer_path,
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
