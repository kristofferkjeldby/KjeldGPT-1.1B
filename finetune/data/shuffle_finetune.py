"""
Shuffles the Q/A pairs in finetune_corpus.txt before tokenization, so tokenize_finetune.py's
val split (the last val_fraction of the concatenated token stream) ends up a random
sample of pairs rather than whichever passages generate_qa.py happened to process last.

Run from within finetune/data/ (see ../FINETUNE_PARAMS.md):
    cd finetune/data
    python3 shuffle_finetune.py          # finetune_corpus.txt -> finetune_corpus_shuffled.txt
"""

import argparse
import os
import random

EOT = "<|endoftext|>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str,
                         default=os.path.join(os.path.dirname(__file__), "finetune_corpus.txt"))
    parser.add_argument("--out_path", type=str,
                         default=os.path.join(os.path.dirname(__file__), "finetune_corpus_shuffled.txt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        text = f.read()

    pairs = [p for p in text.split(f"{EOT}\n") if p.strip()]
    random.Random(args.seed).shuffle(pairs)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p)
            f.write(f"{EOT}\n")

    print(f"shuffled {len(pairs)} pairs -> {args.out_path}")


if __name__ == "__main__":
    main()
