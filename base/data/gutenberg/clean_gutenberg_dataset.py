"""
Cleans raw Gutenberg text files: strips the Project Gutenberg legal boilerplate that
wraps every book, rejoins hard-wrapped lines, and normalizes the character set down to
plain prose punctuation. Works file-by-file over raw_gutenberg/<id>.txt (one file per
book, ~30,800 of them).

Run from within base/data/gutenberg/ (see ../../BASE_PARAMS.md):
    cd base/data/gutenberg
    python clean_gutenberg_dataset.py                # clean every file in raw_gutenberg/
    python clean_gutenberg_dataset.py --limit 20      # clean only the first 20 (smoke test)
"""

import argparse
import os
import re
import time

# Gutenberg wraps every book in a "START OF ... EBOOK" / "END OF ... EBOOK" marker pair,
# followed by ~30-50KB of near-identical legal boilerplate. Keep only the text between
# the markers so the model never sees (and doesn't memorize) that repeated legal text.
GUTENBERG_BOOK = re.compile(
    r"START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\n(.*?)"
    r"END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
    re.S | re.I,
)

# Gutenberg's plain-text files are hard-wrapped to a fixed line width (a relic of the
# printed page). A genuine new paragraph never starts with a lowercase letter, but a line
# that got wrapped mid-sentence continues one -- that's what tells a wrap apart from a
# real paragraph break, since newline *count* alone can't (some sources put a blank line
# after every wrapped line, not just real paragraph breaks).
HYPHEN_WRAP = re.compile(r"-\n+")
NEWLINE_RUN = re.compile(r"\n+")

# Keep letters/digits, common prose punctuation, and whitespace; drop everything else
# (curly quotes, em-dashes, accented letters, footnote symbols, etc.) rather than feed
# the model one-off vocab entries for rare characters. Includes contractions and
# sentence-ending punctuation alongside letters/digits/space.
CHARSET_FILTER = re.compile(r"[^a-zA-Z0-9'\".,!?;:()\-\n ]")


def strip_gutenberg_boilerplate(text):
    books = GUTENBERG_BOOK.findall(text)
    if not books:
        return text
    return "\n\n".join(books)


def dewrap(text):
    text = HYPHEN_WRAP.sub("", text)  # "beauti-\nful" -> "beautiful": a hyphenated word
                                       # split at the wrap is never a real paragraph break
    chunks = [c.strip(" \t") for c in NEWLINE_RUN.split(text)]
    out = [chunks[0]]
    for chunk in chunks[1:]:
        if chunk[:1].islower():
            out.append(" " + chunk)     # continues the previous sentence
        else:
            out.append("\n\n" + chunk)  # genuine paragraph break
    return "".join(out)


def clean_text(text):
    text = strip_gutenberg_boilerplate(text)
    text = dewrap(text)
    text = CHARSET_FILTER.sub("", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="raw_gutenberg")
    parser.add_argument("--clean_dir", type=str, default="clean_gutenberg")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N files (smoke test)")
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
    processed = 0
    skipped = 0

    for i, fname in enumerate(filenames, 1):
        src = os.path.join(raw_dir, fname)
        dst = os.path.join(clean_dir, fname)
        if os.path.exists(dst):
            skipped += 1
            continue

        with open(src, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        cleaned = clean_text(text)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(cleaned)

        total_before += len(text)
        total_after += len(cleaned)
        processed += 1

        if i % 500 == 0 or i == len(filenames):
            elapsed = time.time() - t0
            print(f"[{i}/{len(filenames)}] processed={processed} skipped={skipped} "
                  f"{elapsed:.0f}s elapsed", flush=True)

    print(f"\ndone. processed {processed} files, skipped {skipped} already-cleaned files.")
    if total_before:
        pct = 100 * (1 - total_after / total_before)
        print(f"chars: {total_before:,} -> {total_after:,} ({pct:.1f}% dropped as "
              f"boilerplate/disallowed characters)")


if __name__ == "__main__":
    main()
