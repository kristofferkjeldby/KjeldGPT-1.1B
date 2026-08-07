"""
REPL for the finetuned Q/A model -- parallel to completion.py, but wraps whatever
you type in the "Context: ...\\nQuestion: ...\\nAnswer:" template the model was
finetuned on (see finetune/FINETUNE_PARAMS.md), rather than treating your input as raw
prose to continue. Each prompt is independent -- no conversation history carries over
between turns.

    python3 chat.py
    python3 chat.py --checkpoint finetune/checkpoints/kjeldchat_v5.pt  # an earlier run
    python3 chat.py --length 200 --temperature 0.8 --top_k 50
    python3 chat.py --no-speak      # skip reading answers aloud
    python3 chat.py --no-context    # closed-book: no Context, just the question
    python3 chat.py --debug         # start with prompt/context debug info shown

--context/--no-context, --speak/--no-speak, and --debug/--no-debug (defaults: context
on, speak and debug off) set the starting state of the same three things the
mid-session toggle commands below control.

Mid-session, without restarting (which would mean reloading the model/passage index
from scratch): "context"/"no_context" toggles Context retrieval, "speak"/"no_speak"
toggles speech, on top of whatever --context/--speak the session started with. Only
toggles use of whatever was already loaded at startup -- if a resource wasn't loaded
in the first place (--no-context/--no-speak at startup, or no speech backend on this
platform), the corresponding command reports that instead of loading it lazily.
"debug"/"no_debug" toggles printing each turn's prompt token count and (when Context
is on) the actual retrieved passage, right before the answer streams in.

RAG (retrieval-augmented generation): the model was finetuned open-book -- trained to
answer from a Context passage handed to it, rather than recall facts from its own thin
1.1B-param memory. So before each question is sent to the model, it's first embedded
with the same model used to build the passage index (see rag/rag_retrieve.py and
rag/embed_passages.py) and the closest-matching passage (Wikipedia-only -- see
rag_retrieve.py's module docstring) is retrieved and prepended as that Context. This is the
actual fix for hallucination -- without it, this checkpoint would be answering blind
despite being trained for an open-book format. Falls back to closed-book (no Context)
with a warning if the passage index isn't present locally.

The bi-encoder's top-10 candidates are then rescored by a cross-encoder
(rag/rag_rerank.py, on by default -- --no-rerank disables it), which reads question and
passage together instead of comparing two independent embeddings, and so is much better
at telling the specifically-right passage from a merely topically-close one.

Retrieval finds the closest passage by topical overlap, not relevance -- a question
like "What is your name?" reliably retrieves *something* about names, often some
unrelated person's Wikipedia biography with nothing to do with the person asking.
Rather than assert that as fact, --min_context_score (default 2.0, on the reranker's
logit scale) discards the retrieved passage and falls back to closed-book when the
score is below the floor; "debug"/"no_debug" shows the score and used/skipped status
for every turn. With --no-rerank, scores are bi-encoder cosine similarities instead and
the floor must be given on that scale (--min_context_score 0.55).

Answers are read aloud by default -- offline, no network, cross-platform. On macOS,
shells out to the built-in `say` command (a deliberately robotic voice, Ralph/Fred/
Zarvox/Trinoids, if installed); pyttsx3's macOS driver was confirmed unreliable for
repeated utterances (see make_speech_controller's docstring), so `say` is used there
instead. Elsewhere (Windows/Linux), uses pyttsx3 (pip install pyttsx3, wraps SAPI5/
espeak), which doesn't have that problem. Missing/unavailable backend degrades to a
warning, not a crash. Speech from one answer never blocks the next question -- typing
a new one immediately stops whatever's left of the previous answer's narration.

Type "quit" or Ctrl-C to exit.
"""

import argparse
import itertools
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time

import torch
from tokenizers import Tokenizer

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# model.py lives alongside this script at the repo root; rag_retrieve.py lives in rag/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag"))
from model import GPT, GPTConfig
from rag_retrieve import DEFAULT_INDEX_DIR, Retriever
from rag_rerank import Reranker

# Same display cleanup as completion.py -- the byte-level tokenizer can emit rarely-seen
# control characters, and collapsing linebreaks/doubled spaces keeps the terminal output
# readable. Answers are shorter and more uniform than base-model prose, but the same
# cleanup is harmless here.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # keep \n (\x0a) and \t
LINEBREAKS = re.compile(r"\s*\n\s*")
DOUBLE_SPACES = re.compile(r"[ \t]{2,}")

PROMPT_COLOR = "\033[36m"  # cyan
HEADER_COLOR = "\033[33m"  # yellow
THINKING_COLOR = "\033[2m"  # dim -- the generation spinner and [debug] lines
                            # itself (kept off-screen, see the retrieval call site)
RESET = "\033[0m"

QA_TEMPLATE = "Question: {question}\nAnswer:"
RAG_TEMPLATE = "Context: {context}\nQuestion: {question}\nAnswer:"

# Mid-session toggles (see main()'s REPL loop) -- (resource, turn_on) per command.
# Named to match the --context/--no-context (etc.) startup flags below, just without
# the leading dashes a REPL command has no use for.
TOGGLE_COMMANDS = {
    "context": ("retrieval", True),
    "no_context": ("retrieval", False),
    "speak": ("speech", True),
    "no_speak": ("speech", False),
    "debug": ("debug", True),
    "no_debug": ("debug", False),
}

SPINNER_FRAMES = "|/-\\"


def get_first_chunk_with_spinner(prefix, iterator):
    """prefix (e.g. "Answer: ") is already on screen; this animates a spinner right
    after it -- "Answer: [thinking /]" -- on a background thread pulling the first item
    from iterator, since there's real, unpredictable latency (the model's first
    forward pass) before the first token is ready. Clears the spinner back down to
    just prefix before returning, so the caller's normal streaming loop picks up from a
    clean line."""
    result = {}

    def target():
        result["value"] = next(iterator, None)

    thread = threading.Thread(target=target)
    thread.start()
    i = 0
    suffix_width = 0
    while thread.is_alive():
        suffix = f"[thinking {SPINNER_FRAMES[i % len(SPINNER_FRAMES)]}]"
        suffix_width = max(suffix_width, len(suffix))
        print(f"\r{prefix}{THINKING_COLOR}{suffix}{RESET}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    thread.join()
    print(f"\r{prefix}" + " " * suffix_width + f"\r{prefix}", end="", flush=True)
    return result["value"]

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


# Same backup as completion.py: if EOT never comes (a still-lightly-finetuned checkpoint
# doesn't reliably produce it every time), watch for a sentence-ending period once only
# TAIL_FRACTION of the token budget remains and cut the reply there instead of
# mid-thought; if none shows up before the budget runs out, trail off with "..." rather
# than stopping on a half-written sentence.
SENTENCE_END = re.compile(r"\.")
TAIL_FRACTION = 0.2

# Preference-ordered list of macOS `say` voices to use (first installed one wins).
# Exact names as `say -v ?` lists them (case-sensitive for the -v flag).
MACOS_VOICES = ("Ralph", "Fred", "Zarvox", "Trinoids")
ROBOTIC_RATE = 170  # pyttsx3 default is ~200 wpm -- used on non-macOS platforms, where
                    # SAPI5/espeak have no robotic-sounding named voice to reach for

# Splits the growing answer buffer into speakable chunks as soon as they're ready
# (punctuation followed by whitespace -- a period with nothing after it yet might just
# be mid-generation, e.g. "Mr." or an unfinished decimal), so each chunk can be shown
# and queued for speech immediately instead of waiting for an entire sentence -- let
# alone the whole multi-sentence answer -- to finish generating before anything is
# said or shown. Includes clause-level punctuation (,;:) in addition to sentence
# enders (.!?) so speech starts sooner and follows natural pauses within a sentence.
SENTENCE_BOUNDARY = re.compile(r"[.!?,;:]+(\s+)")

# Hard fallback for a run of words with no punctuation at all (run-on generation) --
# without this, speech would wait indefinitely for a boundary that might not come for
# a while, instead of starting to talk.
MAX_CHUNK_WORDS = 10


def find_macos_voice():
    """First installed voice from MACOS_VOICES (in preference order), or None if `say`
    isn't available or none of them are installed."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    installed = {line.split()[0] for line in out.splitlines() if line.strip()}
    return next((v for v in MACOS_VOICES if v in installed), None)


class SpeechController:
    """Speaks queued sentences one at a time on a dedicated background thread, and can
    immediately abort whatever's currently playing plus drop anything still queued --
    used to cut off the previous answer's speech the moment a new question is
    submitted, rather than making the user wait for it to finish before they can even
    type.

    Takes three small platform-specific functions rather than hardcoding a backend:
      start(text) -> handle   launches the utterance, returns immediately
      wait(handle)             blocks (in the worker thread) until it finishes
      stop(handle)              aborts it (called from the main thread, mid-wait)
    """

    def __init__(self, start, wait, stop):
        self._start = start
        self._wait = wait
        self._stop = stop
        self._queue = queue.Queue()
        self._current = None
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            sentence = self._queue.get()
            with self._lock:
                self._current = self._start(sentence)
            self._wait(self._current)
            with self._lock:
                self._current = None
            self._queue.task_done()

    def say(self, sentence):
        self._queue.put(sentence)

    def stop(self):
        """Drops everything still queued and aborts whatever's currently playing."""
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            pass
        with self._lock:
            if self._current is not None:
                self._stop(self._current)


def make_speech_controller():
    """Returns a SpeechController for this OS, or None if no speech backend is usable
    -- lets --speak degrade to a warning instead of a crash.

    macOS: shells out to the built-in `say` command instead of pyttsx3. pyttsx3's
    macOS driver (nsss) only produces sound for the *first* utterance in a process --
    every engine.say()+runAndWait() call after that returns instantly and silently. A
    fresh `say` subprocess per sentence has no persistent engine state to go stale, and
    reliably speaks every call. Each subprocess.Popen handle can also be .terminate()'d
    directly, which is what makes "stop the previous answer's speech" possible at all.

    Elsewhere (Windows/Linux): pyttsx3 wraps SAPI5/espeak, neither of which is known
    to share that nsss-specific bug, so it's used directly rather than reaching for an
    OS binary that may not exist there. engine.stop() plays the same role as
    Popen.terminate() there.
    """
    if platform.system() == "Darwin":
        voice = find_macos_voice()

        def start(text):
            return subprocess.Popen(["say"] + (["-v", voice] if voice else []) + [text])

        def wait(proc):
            proc.wait()

        def stop(proc):
            proc.terminate()

        return SpeechController(start, wait, stop)

    if pyttsx3 is None:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", ROBOTIC_RATE)
    except Exception:
        return None

    def start(text):
        engine.say(text)
        return engine

    def wait(engine_handle):
        engine_handle.runAndWait()

    def stop(engine_handle):
        engine_handle.stop()

    return SpeechController(start, wait, stop)


def stream_reply(model, idx, tokenizer, eot_id, max_new_tokens, temperature, top_k, repetition_penalty,
                  no_penalty_ids=None, penalize_from=0):
    """Same re-decode-the-growing-suffix approach as completion.py's stream_reply.
    Primary stop is the EOT token, since (unlike the base model) this one was
    explicitly trained to emit EOT right after an answer -- letting the loop run to
    max_new_tokens instead would just append prompt-shaped text for the next
    (nonexistent) turn. Backup stop is completion.py's sentence-boundary/"..." logic,
    for the case where EOT doesn't show up."""
    tail_start = int(max_new_tokens * (1 - TAIL_FRACTION))
    reply_ids = []
    printed = ""
    for i, token in enumerate(model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                                              top_k=top_k, repetition_penalty=repetition_penalty,
                                              no_penalty_ids=no_penalty_ids, penalize_from=penalize_from)):
        token_id = token.item()
        if token_id == eot_id:
            return
        reply_ids.append(token_id)
        full_text = CONTROL_CHARS.sub("", tokenizer.decode(reply_ids))
        full_text = LINEBREAKS.sub(" ", full_text)
        full_text = DOUBLE_SPACES.sub(" ", full_text)
        # A byte-level BPE token can be part of a multi-byte UTF-8 character (an em
        # dash, curly quote, ellipsis, ...) that isn't complete yet -- decode() then
        # emits U+FFFD in its place until the completing token arrives. A *trailing*
        # U+FFFD might still resolve with the next token, so withhold it rather than
        # yielding it (once yielded, `printed` has already advanced past it and
        # nothing re-visits that position). One that's NOT trailing -- i.e. more text
        # has since been decoded after it -- had its chance and is permanently
        # unresolvable (e.g. repetition_penalty blocking the one token that would've
        # completed it): drop it rather than show a stray "�".
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
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune",
                                               "checkpoints", "kjeldchat_v8.pt"))
    parser.add_argument("--tokenizer", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "base", "data", "tokenizer", "tokenizer.json"))
    parser.add_argument("--length", type=int, default=100, help="max tokens generated per reply")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                         help="discourages re-picking already-used tokens (1.0 = off, "
                              "typical range 1.1-1.3, higher = stronger)")
    parser.add_argument("--speak", action=argparse.BooleanOptionalAction, default=False,
                         help="read answers aloud (macOS `say`, or pyttsx3 elsewhere). "
                              "Off by default -- pass --speak to enable. Same starting "
                              "state as the mid-session \"speak\"/\"no_speak\" toggle")
    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True,
                         help="embed each question and prepend the closest-matching "
                              "passage from the training corpus as Context (see "
                              "rag_retrieve.py) -- this is what the model was actually "
                              "finetuned to expect. --no-context for closed-book "
                              "(question only, no Context). Same starting state as the "
                              "mid-session \"context\"/\"no_context\" toggle")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False,
                         help="start with each turn's prompt token count and (when "
                              "Context is on) the retrieved passage printed before the "
                              "answer. Same starting state as the mid-session "
                              "\"debug\"/\"no_debug\" toggle")
    parser.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR,
                         help="directory containing embeddings.npy/passages.txt/meta.json "
                              "from data/embed_passages.py")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True,
                         help="rerank the bi-encoder's top-10 candidates with rag/rag_rerank.py's "
                              "cross-encoder before picking the best passage -- see its "
                              "module docstring. --no-rerank for the plain bi-encoder path "
                              "(then also pass --min_context_score 0.55, since the "
                              "cross-encoder's score is on a different scale)")
    parser.add_argument("--min_context_score", type=float, default=2.0,
                         help="score floor below which the retrieved passage is discarded "
                              "and the turn falls back to closed-book, rather than handing "
                              "over a merely topically-similar (not actually relevant) "
                              "passage as Context. 2.0 is calibrated for --rerank's "
                              "cross-encoder score scale (see rag/rag_rerank.py) -- pass 0.55 "
                              "instead if using --no-rerank (plain bi-encoder cosine "
                              "similarity, a completely different scale)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer from {args.tokenizer} ...", flush=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    no_penalty_ids = continuation_byte_token_ids(tokenizer)

    speech = None
    if args.speak:
        print("Initializing speech backend ...", flush=True)
        speech = make_speech_controller()
        if speech is None:
            print(f"{HEADER_COLOR}--speak requested but no speech backend is available "
                  f"on this platform -- `pip install pyttsx3` to enable. Continuing "
                  f"without speech.{RESET}")

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

    retriever = None
    if args.context:
        if os.path.exists(os.path.join(args.index_dir, "meta.json")):
            print(f"Loading passage index from {args.index_dir} (embedding model load "
                  f"plus a one-time byte-offset scan if not already cached -- can take "
                  f"a minute) ...", flush=True)
            reranker = None
            if args.rerank:
                print("Loading cross-encoder reranker ...", flush=True)
                reranker = Reranker()
            retriever = Retriever(args.index_dir, reranker=reranker)
            print(f"  {retriever.index.meta['num_passages']:,} passages indexed", flush=True)
        else:
            print(f"{HEADER_COLOR}--context requested but no passage index found at "
                  f"{args.index_dir} -- run data/embed_passages.py first. Continuing "
                  f"closed-book, no Context.{RESET}")

    # Runtime toggles (see TOGGLE_COMMANDS) -- start matching whatever --context/
    # --speak/--debug loaded, but can be flipped mid-session without reloading anything.
    retrieval_on = retriever is not None
    speech_on = speech is not None
    debug_on = args.debug

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    os.system("clear")
    header = [
        "KjeldChat 1.1B",
        f"Loaded checkpoint: step {ckpt['iter']}, val_loss {ckpt['val_loss']:.4f}, "
        f"{n_params:.2f}M params, device={device}",
        f"Retrieval: {'ON, ' + format(retriever.index.meta['num_passages'], ',') + ' passages' if retriever else 'OFF'}",
        'Type a question, "context"/"no_context", "speak"/"no_speak" or "debug"/"no_debug" '
        'to toggle, or "quit" to exit.',
    ]
    header_text = "\n".join(header)
    print(f"{HEADER_COLOR}{header_text}{RESET}")
    print(f"{HEADER_COLOR}{'-' * max(len(line) for line in header)}{RESET}")
    print()

    while True:
        try:
            user_input = input(f"{PROMPT_COLOR}Question: {RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            if speech is not None:
                speech.stop()
            break

        if user_input.strip().lower() in ("quit", "exit"):
            if speech is not None:
                speech.stop()
            break
        if not user_input.strip():
            continue

        toggle = TOGGLE_COMMANDS.get(user_input.strip().lower())
        if toggle is not None:
            print()
            resource, turn_on = toggle
            if resource == "retrieval":
                if retriever is None:
                    print(f"{HEADER_COLOR}No passage index was loaded at startup -- "
                          f"restart without --no-context to enable Context.{RESET}")
                else:
                    retrieval_on = turn_on
                    print(f"{HEADER_COLOR}Context: {'ON' if retrieval_on else 'OFF'}{RESET}")
            elif resource == "speech":
                if speech is None:
                    print(f"{HEADER_COLOR}No speech backend was loaded at startup -- "
                          f"restart with --speak to enable it.{RESET}")
                else:
                    speech_on = turn_on
                    if not speech_on:
                        speech.stop()  # silence whatever's currently playing right away
                    print(f"{HEADER_COLOR}Speak: {'ON' if speech_on else 'OFF'}{RESET}")
            else:
                debug_on = turn_on
                print(f"{HEADER_COLOR}Debug: {'ON' if debug_on else 'OFF'}{RESET}")
            print()
            continue

        print()

        # A new question just came in -- cut off whatever's left of the previous
        # answer's speech (queued sentences and whatever's currently playing) instead
        # of making the user wait for it to finish. This is also why the speech
        # branch below never blocks for the *current* answer either: control returns
        # to this prompt immediately, and it's this call, next time round, that stops
        # it if the user doesn't wait.
        if speech is not None:
            speech.stop()

        context, score = None, None
        if retriever is not None and retrieval_on:
            context, score = retriever.best_passage(user_input.strip())
            if score < args.min_context_score:
                # Merely topically similar, not actually relevant (see --min_context_score's
                # help) -- fall back to closed-book rather than assert this as fact.
                context = None

        if context is not None:
            prompt = RAG_TEMPLATE.format(context=context, question=user_input.strip())
        else:
            prompt = QA_TEMPLATE.format(question=user_input.strip())
        ids = tokenizer.encode(prompt).ids

        if debug_on:
            print(f"{THINKING_COLOR}[debug] {len(ids)} prompt tokens{RESET}")
            if score is not None:
                status = "used" if context is not None else f"skipped, below --min_context_score {args.min_context_score}"
                print(f"{THINKING_COLOR}[debug] retrieval score: {score:.3f} ({status}){RESET}")
            if context is not None:
                print(f"{THINKING_COLOR}[debug] context: {context}{RESET}")
            print()  # separates the debug block from Answer -- the blank line after
                     # Question above already separates it when debug is off

        idx = torch.tensor([ids], dtype=torch.long, device=device)
        answer_prefix = f"{PROMPT_COLOR}Answer: {RESET}"
        print(answer_prefix, end="", flush=True)

        reply_iter = iter(stream_reply(model, idx, tokenizer, eot_id, args.length, args.temperature,
                                        args.top_k, args.repetition_penalty, no_penalty_ids,
                                        penalize_from=len(ids)))
        first = get_first_chunk_with_spinner(answer_prefix, reply_iter)
        reply = itertools.chain([first], reply_iter) if first is not None else iter(())

        if speech is None or not speech_on:
            # No speech in play -- print live as each chunk is generated, exactly as
            # before. One-chunk lookahead (print the previous chunk, not the current
            # one) so the truly-last chunk can be rstripped before printing -- models
            # often end generation on a trailing space, which without this would show
            # up as a stray space after the final word.
            first_chunk = True
            pending = None
            for chunk in reply:
                if first_chunk:
                    chunk = chunk.lstrip(" ")  # "Answer: " already supplied the space
                    first_chunk = False
                if pending is not None:
                    print(pending, end="", flush=True)
                pending = chunk
            if pending is not None:
                print(pending.rstrip(), end="", flush=True)
            print("\n")
        else:
            # Stream live exactly like the no-speech branch above -- same token-by-
            # token print, no "generating..." placeholder -- and additionally queue
            # each completed sentence for speech (speech.say() never blocks) the
            # moment its boundary is seen. Because printing never waits on that queue,
            # the *next* sentence keeps streaming to the screen while the previous one
            # is still playing in the speech worker thread -- and neither blocks
            # returning to the prompt below for the next question.
            first_chunk = True
            buf = ""
            pending = None
            for chunk in reply:
                if first_chunk:
                    chunk = chunk.lstrip(" ")  # "Answer: " already supplied the space
                    first_chunk = False
                if pending is not None:
                    print(pending, end="", flush=True)
                pending = chunk
                buf += chunk

                m = SENTENCE_BOUNDARY.search(buf)
                if m:
                    sentence, buf = buf[:m.end()], buf[m.end():]
                    speech.say(sentence)
                elif len(buf.split()) >= MAX_CHUNK_WORDS:
                    cut = buf.rfind(" ")
                    if cut != -1:  # keep any trailing partial word in buf
                        sentence, buf = buf[:cut + 1], buf[cut + 1:]
                        speech.say(sentence)

            if pending is not None:
                print(pending.rstrip(), end="", flush=True)
            if buf:
                speech.say(buf)
            print("\n")


if __name__ == "__main__":
    main()
