"""
One-time corpus migration: re-applies the corrected clip_to_last_sentence() (see
generate_qa.py) to the Context field of every already-generated Q/A pair, so existing
passages stop ending mid-word/mid-sentence (e.g. "...at which boil") the way the old
char/token-only truncation left them. Only the Context field is touched -- Question
and Answer are left exactly as Claude generated them, so this needs no API calls and
costs nothing to re-run, unlike regenerating the corpus from scratch would.

Safe because the dropped text is always the trailing PARTIAL sentence (the literal
cutoff artifact) -- never a complete sentence, so nothing a generated Q/A pair could
have been grounded in gets removed.

Already applied to the corpus in the repo (it is what separates the v5 corpus from
v4's -- see ../FINETUNE_PARAMS.md's "Round 4 results"). Kept as the record of what was
changed, and because it is idempotent: an already-clipped Context is left alone.

Re-run it, then shuffle_finetune.py and tokenize_finetune.py, if the corpus is ever
regenerated:
    cd finetune/data
    python3 fix_context_truncation.py
"""

import os
import re

from generate_qa import clip_to_last_sentence

EOT = "<|endoftext|>"
CONTEXT_RE = re.compile(r"^Context: (.*?)\nQuestion: ", re.DOTALL)

FILES = [
    "finetune_corpus_context.txt",
    "finetune_corpus_context_false_premise.txt",
    "finetune_corpus_context_grounding.txt",
]


def fix_file(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    blocks = [b for b in text.split(f"{EOT}\n") if b.strip()]
    fixed_blocks = []
    clipped = 0
    for block in blocks:
        match = CONTEXT_RE.match(block)
        if match is None:
            fixed_blocks.append(block)
            continue
        context = match.group(1)
        clipped_context = clip_to_last_sentence(context)
        if clipped_context != context:
            clipped += 1
        fixed_blocks.append(block[:match.start(1)] + clipped_context + block[match.end(1):])

    with open(path, "w", encoding="utf-8") as f:
        for b in fixed_blocks:
            f.write(b)
            f.write(f"{EOT}\n")

    print(f"{path}: {clipped}/{len(blocks)} contexts clipped")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    for fname in FILES:
        fix_file(os.path.join(base, fname))


if __name__ == "__main__":
    main()
