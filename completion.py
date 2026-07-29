"""
Simple REPL: type text, the model continues it, streaming the reply token by token as
it's generated (chat-style) rather than waiting for the whole thing and printing it all
at once. Each prompt is independent -- no conversation history carries over between
turns, since this is a base language model completing whatever text it's given, not a
chat model with a notion of prior turns.

Lives at the repo root alongside chat.py, which drives the finetuned Q/A + RAG model
instead -- this one talks to the base model, a test script for base_train.py, not the
deliverable.

    python completion.py                          # run from the repo root
    python completion.py --checkpoint base/checkpoints/kjeldgpt.pt --tokenizer base/data/tokenizer/tokenizer.json
    python completion.py --length 200 --temperature 0.8 --top_k 50

Type "quit" or Ctrl-C to exit.
"""

import argparse
import os
import re
import time

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

# The byte-level tokenizer's vocab covers all 256 possible byte values (so it can
# represent any input), including control characters the data-cleaning scripts'
# charset filters stripped out of the training text -- e.g. BEL (\x07), which terminals
# render as an audible beep. The model rarely emits these (they're barely trained on),
# but "rarely" isn't "never", especially early in training, so strip them at display time.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # keep \n (\x0a) and \t

# The base model reproduces Gutenberg/Wikipedia's line-wrapping and paragraph breaks
# verbatim, which reads as random mid-sentence linebreaks once printed to a terminal
# that already wraps -- collapse any run of whitespace containing a newline to a
# single space so the reply reads as one continuous, terminal-wrapped paragraph.
LINEBREAKS = re.compile(r"\s*\n\s*")

# Collapsing linebreaks to spaces above can itself create doubled-up spaces (e.g. a line
# that already ended in a space, followed by an indented next line), on top of any the
# model emits directly -- collapse any run of spaces/tabs to one.
DOUBLE_SPACES = re.compile(r"[ \t]{2,}")

# Base-model generation has no notion of "wrapping up" -- left alone it just stops
# mid-sentence wherever the token budget runs out. Once only 20% of the budget remains,
# start watching for a sentence-ending period and cut the reply there instead of
# mid-thought; if none shows up before the budget is exhausted, trail off with "..."
# rather than stopping on a half-written sentence.
SENTENCE_END = re.compile(r"\.")
TAIL_FRACTION = 0.2

HEADER_COLOR = "\033[33m"  # yellow -- startup banner
LABEL_COLOR = "\033[36m"  # cyan -- "Input:"/"Completion:" labels, same as chat.py's PROMPT_COLOR
INPUT_COLOR = "\033[37m"  # white -- text typed at the Input: prompt
ECHO_COLOR = "\033[33m"  # yellow -- the input echoed back on the Completion: line
COMPLETION_COLOR = "\033[37m"  # white -- the generated completion itself
RESET = "\033[0m"


def continuation_byte_token_ids(tokenizer):
    """Token ids whose solo decode() isn't a complete character on its own (contains
    U+FFFD) -- raw UTF-8 continuation bytes that a multi-byte character (accented
    letters, curly quotes, em dashes, ...) gets split across, and whose specific byte
    values are shared across many unrelated characters. Passed to model.generate() as
    no_penalty_ids so repetition_penalty never suppresses them -- see its docstring for
    why that suppression happens and why it's wrong for tokens like these."""
    vocab_size = tokenizer.get_vocab_size()
    ids = [i for i in range(vocab_size) if "�" in tokenizer.decode([i])]
    return torch.tensor(ids, dtype=torch.long)


def stream_reply(model, idx, tokenizer, max_new_tokens, temperature, top_k, repetition_penalty,
                  no_penalty_ids=None):
    """Yields each newly-decodable chunk of text as tokens are sampled. Re-decodes the
    whole reply-so-far on every new token rather than decoding each token id in
    isolation -- a single byte-level BPE token can be part of a multi-byte UTF-8
    character, so decoding it alone can produce garbage or a transient replacement
    character right up until the character-completing token arrives. A *trailing*
    U+FFFD might still resolve with the next token, so it's withheld rather than
    yielded (once yielded, `printed` has already advanced past it and nothing
    re-visits that position). One that's NOT trailing -- more text has since been
    decoded after it -- had its chance and is permanently unresolvable (e.g.
    repetition_penalty blocking the one token that would've completed it): dropped
    rather than shown as a stray "�".

    Also stops the reply at a sentence boundary rather than an arbitrary token cutoff --
    see SENTENCE_END above."""
    tail_start = int(max_new_tokens * (1 - TAIL_FRACTION))
    reply_ids = []
    printed = ""
    for i, token in enumerate(model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                                              top_k=top_k, repetition_penalty=repetition_penalty,
                                              no_penalty_ids=no_penalty_ids)):
        reply_ids.append(token.item())
        full_text = CONTROL_CHARS.sub("", tokenizer.decode(reply_ids))
        full_text = LINEBREAKS.sub(" ", full_text)
        full_text = DOUBLE_SPACES.sub(" ", full_text)
        safe_text = DOUBLE_SPACES.sub(" ", full_text.rstrip("�").replace("�", ""))
        new_text = safe_text[len(printed):]
        if not new_text:
            continue
        printed = safe_text

        if i >= tail_start:
            m = SENTENCE_END.search(new_text)
            if m:
                yield new_text[:m.end()]
                return

        yield new_text
    else:
        yield "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="base/checkpoints/kjeldgpt.pt")
    parser.add_argument("--tokenizer", type=str, default="base/data/tokenizer/tokenizer.json")
    parser.add_argument("--length", type=int, default=100, help="tokens generated per reply")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.3,
                         help="discourages re-picking already-used tokens (1.0 = off, "
                              "typical range 1.1-1.3, higher = stronger)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer from {args.tokenizer} ...", flush=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    no_penalty_ids = continuation_byte_token_ids(tokenizer)

    # The checkpoint is multi-GB -- this is the slow part (can take tens of seconds),
    # so it's the one step most likely to look like a hang without a status line.
    print(f"Loading checkpoint from {args.checkpoint} (this can take a while for a "
          f"multi-GB file) ...", flush=True)
    t0 = time.time()
    torch.serialization.add_safe_globals([GPTConfig])
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print(f"Building model on {device} ...", flush=True)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    os.system("clear")
    header = [
        "KjeldGPT 1.1B",
        f"Loaded checkpoint: step {ckpt['iter']}, val_loss {ckpt['val_loss']:.4f}, "
        f"{n_params:.2f}M params, device={device}",
        'Type "quit" to exit.',
    ]
    header_text = "\n".join(header)
    print(f"{HEADER_COLOR}{header_text}{RESET}")
    print(f"{HEADER_COLOR}{'-' * max(len(line) for line in header)}{RESET}")
    print()

    while True:
        try:
            user_input = input(f"{LABEL_COLOR}Input: {INPUT_COLOR}")
        except (EOFError, KeyboardInterrupt):
            print(RESET)
            break

        print(RESET, end="")
        if user_input.strip().lower() in ("quit", "exit"):
            break

        ids = tokenizer.encode(user_input).ids
        if not ids:
            continue

        idx = torch.tensor([ids], dtype=torch.long, device=device)
        print()
        print(f"{LABEL_COLOR}Completion: {RESET}{ECHO_COLOR}{user_input}{RESET}{COMPLETION_COLOR}",
              end="", flush=True)
        for chunk in stream_reply(model, idx, tokenizer, args.length, args.temperature, args.top_k,
                                   args.repetition_penalty, no_penalty_ids):
            print(chunk, end="", flush=True)
        print(f"{RESET}\n")


if __name__ == "__main__":
    main()
