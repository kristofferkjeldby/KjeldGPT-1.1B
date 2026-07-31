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
    parser.add_argument("--group_by_context", action="store_true",
                         help="shuffle whole Context groups instead of individual pairs, "
                              "keeping every pair that shares a Context adjacent. Pass this "
                              "together with tokenize_finetune.py's --group_val_by_context "
                              "so no Context has pairs on both sides of the train/val "
                              "split. Matters whenever one passage carries several pairs "
                              "(generate_qa.py writes 2, generate_discrimination_qa.py up "
                              "to 4): split pair-wise, near-identical siblings land in both "
                              "train and val, val loss reads better than it should, and the "
                              "early-stopping signal the whole recipe leans on gets weaker "
                              "the more contrastive the corpus is")
    args = parser.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        text = f.read()

    pairs = [p for p in text.split(f"{EOT}\n") if p.strip()]
    rng = random.Random(args.seed)
    if args.group_by_context:
        groups = {}
        for p in pairs:
            # First line of each block is "Context: ...", the whole passage on one
            # physical line (generate_qa.py's truncate_to_token_budget guarantees it).
            groups.setdefault(p.split("\n", 1)[0], []).append(p)
        ordered = list(groups.values())
        rng.shuffle(ordered)
        pairs = [p for group in ordered for p in group]
        print(f"grouped {len(pairs)} pairs into {len(ordered)} Context groups "
              f"({len(pairs) / max(len(ordered), 1):.1f} pairs per Context)")
    else:
        rng.shuffle(pairs)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p)
            f.write(f"{EOT}\n")

    print(f"shuffled {len(pairs)} pairs -> {args.out_path}")


if __name__ == "__main__":
    main()
