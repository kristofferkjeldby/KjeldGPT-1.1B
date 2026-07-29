"""
Assembles the Hugging Face Space repo for KjeldChat into a staging directory, so the
Space's contents stay generated from this repo rather than hand-maintained in a second
place that silently drifts.

A Space is a flat git repo with app.py at its root, but the app imports chat_gui.py,
chat.py, model.py and rag/'s two retrieval modules -- so those are copied in at the
same relative paths they have here, which is what lets app.py reuse them unmodified.
Nothing large is copied: weights, the passage index and the sentence-transformer
snapshots are all downloaded from the Hub at Space startup (see app.py).

Run from within spaces/:
    cd spaces
    python3 build_space.py                    # -> spaces/build/
    python3 build_space.py --out_dir /tmp/x   # somewhere else

Then push the result:
    hf upload kristofferkjeldby/KjeldChat-1.1B build . --repo-type space
"""

import argparse
import os
import shutil

SPACES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SPACES_DIR)

# (source path relative to the project root, destination path relative to the Space root)
FILES = [
    ("spaces/app.py", "app.py"),
    ("spaces/requirements.txt", "requirements.txt"),
    ("spaces/README.md", "README.md"),
    ("model.py", "model.py"),
    ("chat.py", "chat.py"),
    ("chat_gui.py", "chat_gui.py"),
    ("rag/rag_retrieve.py", "rag/rag_retrieve.py"),
    ("rag/rag_rerank.py", "rag/rag_rerank.py"),
    ("logo.png", "logo.png"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(SPACES_DIR, "build"),
                         help="staging directory to assemble the Space into")
    args = parser.parse_args()

    # Rebuilt from scratch each time: a stale file left behind from a previous layout
    # would be uploaded and silently shadow the real one.
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)

    for src_rel, dst_rel in FILES:
        src = os.path.join(ROOT, src_rel)
        dst = os.path.join(args.out_dir, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src_rel} -> {dst_rel}")

    print(f"\nassembled {len(FILES)} files into {args.out_dir}")
    print(
        "\nTo deploy:\n"
        "    hf upload kristofferkjeldby/KjeldChat-1.1B "
        f"{os.path.relpath(args.out_dir, os.getcwd())} . --repo-type space\n"
        "\nThe Space must have ZeroGPU selected in its settings -- that's a hardware\n"
        "setting on the Hub, not something the README frontmatter can request.\n"
    )


if __name__ == "__main__":
    main()
