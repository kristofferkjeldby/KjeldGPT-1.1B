"""
Trains a byte-level BPE tokenizer jointly on the combined corpus -- Gutenberg
(clean_gutenberg/) plus Wikipedia (clean_wikipedia/) -- using HuggingFace `tokenizers`
(Rust-backed, fast enough to finish on tens of GB in minutes).

vocab_size defaults to 50257 (GPT-2's own vocab size) -- big enough for GPT-2's full
subword coverage across a corpus this varied (encyclopedic Wikipedia plus Gutenberg's
history/biography/philosophy/adventure books).

Retraining the tokenizer changes every token id, so any checkpoint trained against a
previous tokenizer can no longer be resumed against this one -- this is a fresh-start
vocab, not a compatible update.

Run from within data/ (see ../BASE_PARAMS.md):
    cd data
    python vocab_dataset.py                          # combined corpus, vocab_size=50257
    python vocab_dataset.py --limit 200               # smoke test on a subset first

Requires: pip install tokenizers
"""

import argparse
import os
import time

try:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
except ImportError as e:
    raise SystemExit(
        "the `tokenizers` package is required (pip install tokenizers)"
    ) from e

from tqdm import tqdm

# Marks the boundary between two concatenated books, so the model can learn where one
# document ends and another begins rather than reading the whole corpus as one
# continuous, unrelated stream of text.
EOT_TOKEN = "<|endoftext|>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, action="append", default=None,
                         help="dir of cleaned .txt files to train on -- pass multiple "
                              "times to override the default (Gutenberg + Wikipedia)")
    parser.add_argument("--vocab_size", type=int, default=50257,
                         help="target vocab size, 256 base byte tokens + learned merges")
    parser.add_argument("--min_frequency", type=int, default=2,
                         help="minimum pair frequency to learn a merge (HF BpeTrainer default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only train on the first N cleaned files (smoke test)")
    parser.add_argument("--out", type=str,
                         default=os.path.join("tokenizer", "tokenizer.json"),
                         help="output path for the trained tokenizer.json")
    args = parser.parse_args()
    clean_dirs = args.clean_dir or [
        os.path.join("gutenberg", "clean_gutenberg"),
        os.path.join("wikipedia", "clean_wikipedia"),
    ]

    for clean_dir in clean_dirs:
        if not os.path.isdir(clean_dir):
            raise FileNotFoundError(f"{clean_dir} not found -- run gutenberg/clean_gutenberg_dataset.py "
                                     f"or wikipedia/clean_wikipedia_dataset.py first")

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}")

    log(f"listing files in {', '.join(clean_dirs)}")
    paths = []
    for clean_dir in clean_dirs:
        filenames = sorted(os.listdir(clean_dir))
        paths.extend(os.path.join(clean_dir, f) for f in filenames)
    if args.limit:
        paths = paths[: args.limit]
    paths = list(tqdm(paths, desc="collecting files", unit="file"))

    total_bytes = sum(os.path.getsize(p) for p in tqdm(paths, desc="sizing corpus", unit="file"))
    log(f"training corpus: {len(paths):,} files, {total_bytes / 1e9:.2f} GB")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,  # HF prints its own Rust-side progress bars during training
        special_tokens=[EOT_TOKEN],
        initial_alphabet=ByteLevel.alphabet(),  # guarantee all 256 byte values are covered
    )

    log(f"training BPE tokenizer (vocab_size={args.vocab_size}, min_frequency={args.min_frequency}) "
        f"-- this is the slow part, progress bars below")
    tokenizer.train(paths, trainer)

    actual_vocab_size = tokenizer.get_vocab_size()
    log(f"trained tokenizer: {actual_vocab_size} tokens")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tokenizer.save(args.out)
    log(f"saved to {args.out}")


if __name__ == "__main__":
    main()
