"""
Cleans sharded Wikipedia text (raw_wikipedia/shard_*.txt, each shard holding many
articles joined by the literal "<|endoftext|>" marker). Applies the
same charset normalization as clean_gutenberg_dataset.py, but per-article rather than per-file, and
splits/rejoins on the EOT marker instead of stripping Gutenberg boilerplate -- these
articles never had that wrapper to begin with.

Run from within base/data/wikipedia/ (see ../../BASE_PARAMS.md):
    cd base/data/wikipedia
    python clean_wikipedia_dataset.py                    # clean every shard in raw_wikipedia/
    python clean_wikipedia_dataset.py --limit 20          # smoke test on first 20 shards

Requires: pip install tqdm
"""

import argparse
import os
import time

from tqdm import tqdm

EOT_TOKEN = "<|endoftext|>"
SPLIT_MARKER = f"\n\n{EOT_TOKEN}\n\n"

# Same charset as clean_gutenberg_dataset.py -- keep letters/digits, common prose punctuation, and
# whitespace, drop the rest, so Wikipedia and Gutenberg text share one vocabulary instead
# of each contributing one-off characters the other never uses.
import re
CHARSET_FILTER = re.compile(r"[^a-zA-Z0-9'\".,!?;:()\-\n ]")

# A cleaned article that's mostly punctuation/markup residue (infobox fragments, table
# cells) rather than prose isn't worth keeping -- this re-applies the same
# MIN_ARTICLE_CHARS floor post-cleaning, since the charset filter can hollow out an
# article that looked long enough before stripping wiki-table leftovers.
MIN_ARTICLE_CHARS = 150


def clean_article(text):
    return CHARSET_FILTER.sub("", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="raw_wikipedia")
    parser.add_argument("--clean_dir", type=str, default="clean_wikipedia")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N shard files (smoke test)")
    args = parser.parse_args()
    raw_dir = args.raw_dir
    clean_dir = args.clean_dir

    os.makedirs(clean_dir, exist_ok=True)

    filenames = sorted(os.listdir(raw_dir))
    if args.limit:
        filenames = filenames[:args.limit]

    t0 = time.time()
    total_before = 0
    total_after = 0
    articles_in = 0
    articles_out = 0
    processed = 0
    skipped = 0

    for fname in tqdm(filenames, desc="cleaning shards", unit="shard"):
        src = os.path.join(raw_dir, fname)
        dst = os.path.join(clean_dir, fname)
        if os.path.exists(dst):
            skipped += 1
            continue

        with open(src, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        articles = text.split(SPLIT_MARKER)
        cleaned_articles = []
        for article in articles:
            total_before += len(article)
            cleaned = clean_article(article)
            if len(cleaned) < MIN_ARTICLE_CHARS:
                continue
            cleaned_articles.append(cleaned)
            total_after += len(cleaned)

        articles_in += len(articles)
        articles_out += len(cleaned_articles)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(SPLIT_MARKER.join(cleaned_articles))

        processed += 1

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.0f}s. processed {processed} shards, skipped {skipped} "
          f"already-cleaned shards.")
    print(f"articles: {articles_in:,} -> {articles_out:,} "
          f"({articles_in - articles_out:,} dropped as too-short post-clean)")
    if total_before:
        pct = 100 * (1 - total_after / total_before)
        print(f"chars: {total_before:,} -> {total_after:,} ({pct:.1f}% dropped as "
              f"disallowed characters)")


if __name__ == "__main__":
    main()
