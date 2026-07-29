"""
Browser-based chat UI for the finetuned Q/A model -- same pipeline as chat.py (model
load, retrieval + optional reranking, streaming generation), just served through
Gradio's gr.ChatInterface instead of a terminal REPL. Reuses chat.py's prompt
templates/streaming logic directly rather than reimplementing them, so both entry
points stay in sync.

--debug shows each turn's retrieved Context (if any) as a collapsible block under the
answer -- the GUI equivalent of chat.py's "debug" toggle, off by default same as there.

    python3 chat_gui.py                     # opens http://127.0.0.1:7860
    python3 chat_gui.py --no-rerank --min_context_score 0.55
    python3 chat_gui.py --debug             # show retrieved Context under each answer
    python3 chat_gui.py --share             # temporary public Gradio link

Requires: pip install gradio
"""

import argparse
import os
import sys
import time

import gradio as gr
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag"))
from model import GPT, GPTConfig
from rag_retrieve import DEFAULT_INDEX_DIR, Retriever
from rag_rerank import Reranker
from chat import QA_TEMPLATE, RAG_TEMPLATE, continuation_byte_token_ids, stream_reply

THEME = gr.themes.Origin(
    primary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
)

# The header is hand-built HTML rather than logo.png: the mark is drawn as inline SVG so
# it inherits the violet accent below, instead of shipping a second, recoloured copy of
# the amber wordmark that would then have to be kept in sync with it. logo.png is still
# the project's logo everywhere else (README, model cards) -- it just isn't used here.
HEADER_HTML = """
<div id="kjeldchat-header">
  <div class="kc-brand">
    <div class="kc-avatar">
      <!-- Stem and both diagonals are drawn around x=20, the viewBox's centre, so the
           glyph sits centred in the circle once the round caps are accounted for. -->
      <svg width="34" height="34" viewBox="0 0 40 40" fill="none" aria-hidden="true">
        <path d="M14 9v22M14 20.5l11.5-11.5M14 20l12 11" stroke="#fff"
              stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div class="kc-title">KjeldChat</div>
    <div class="kc-badge">1.1B</div>
  </div>
</div>
"""

# Shown in the empty chat area before the first question. Doubles as the place to set
# expectations -- one Q/A format, no conversational memory (see chat.py's templates).
#
# Deliberately attribute-free: Gradio sanitises the placeholder's HTML and strips both
# `id` and `style`, so anything set here is silently dropped (it survives locally but
# not on a Space, which is a nasty way to find out). All of the styling lives in CSS
# instead, keyed on Gradio's own .placeholder class and this markup's child order.
PLACEHOLDER_HTML = """
<div>
  <div>Ask a question</div>
  <div>
    Answers are grounded in retrieved Wikipedia passages.<br>
    Each question is answered independently -- there is no conversational memory.
  </div>
</div>
"""

CSS = """
/* Deliberately no height here. Inside a Space the page runs in an iframe that the parent
   resizes to our reported content height -- pinning html/body to 100vh makes the content
   height equal the frame height, so the frame can grow but never shrink back. Leaving it
   to the content lets the frame settle on the card's actual size. */
html, body {
    overflow: hidden !important;
    height: auto !important; min-height: 0 !important;
    /* Set here, not just on the container: with html/body no longer stretching to the
       viewport, only a background on body propagates to the canvas and covers the area
       below the card. Literal rather than var(--kc-bg), which is defined further in.
       The value matches the Spaces page's own dark background, rgb(11,15,25) -- a darker
       one leaves a visible seam where our iframe starts. */
    background: #0b0f19 !important;
}
[class*="gradio-container"] {
    max-width: 100% !important; width: 100% !important;
    overflow: hidden !important;
    /* Same reason: Gradio's container is a flex child that grows to fill, which would
       keep the reported content height equal to the frame height. */
    height: auto !important; min-height: 0 !important; max-height: none !important;
    flex-grow: 0 !important;
    --input-text-size: var(--chatbot-text-size);
    --kc-bg: #0b0f19;
    --kc-panel: #0d1120;
    --kc-line: rgba(139, 92, 246, 0.22);
    --kc-accent: #7c3aed;
    --kc-accent-2: #a855f7;
    --kc-muted: #8b93ad;
    --background-fill-primary: var(--kc-bg) !important;
    background-color: var(--kc-bg) !important;
}

/* The outer card the whole app sits in -- the rounded, hairline-bordered panel. */
#kjeldchat-wrap {
    max-width: 1000px !important; margin: 24px auto !important; padding: 0 !important;
    gap: 0 !important;
    /* Shrinks with the window, but capped in px. The cap is what makes this safe inside
       a Space: the parent iframe is sized to our content, so a purely viewport-relative
       height feeds back into itself (measured: the frame went 958px then 1038px inside
       an 800px window, and kept climbing). A px ceiling gives that loop a fixed point to
       settle on. 700px also leaves the embed clear of HF's ~50px header on a standard
       window. This cannot be conditioned on being embedded: Gradio 6.20 runs neither
       launch(js=...) nor <script> inside gr.HTML, so there is no hook to detect it. */
    height: min(calc(100vh - 48px), 700px) !important;
    max-height: min(calc(100vh - 48px), 700px) !important;
    /* Floor: below this the transcript is too short to read a reply in, so the card
       stops shrinking. Set well under any realistic desktop window so it only bites on
       a deliberately tiny one. */
    min-height: 420px !important;
    background: var(--kc-panel) !important;
    border: 1px solid var(--kc-line) !important;
    border-radius: 24px !important;
    box-shadow: 0 0 0 1px rgba(0,0,0,.3), 0 24px 60px -20px rgba(88, 28, 135, .45) !important;
    overflow: hidden !important;
}

/* ---- phones --------------------------------------------------------------
   On a phone the Spaces page's own header takes ~119px and the browser's bottom bar
   another ~90px of an 852px viewport, leaving roughly 640px -- a 700px card puts the
   composer below the fold, off-screen. Measured on the live Space at 393x852. The card
   also gives back its side margins, since horizontal room is the scarce thing here. */
@media (max-width: 640px) {
    #kjeldchat-wrap {
        margin: 10px auto !important;
        height: min(calc(100vh - 20px), 560px) !important;
        max-height: min(calc(100vh - 20px), 560px) !important;
        min-height: 320px !important;
        border-radius: 18px !important;
        /* The desktop glow spreads 60px, which on a phone covers most of the visible
           area around the card and reads as a lighter, purple-tinted background. */
        box-shadow: 0 10px 30px -22px rgba(88, 28, 135, .5) !important;
    }
    #kjeldchat-header { padding: 10px 14px 14px !important; }
    #kjeldchat-header .kc-brand { gap: 9px !important; }
    #kjeldchat-header .kc-avatar {
        width: 34px !important; height: 34px !important; flex: 0 0 34px !important;
    }
    #kjeldchat-header .kc-avatar svg { width: 26px !important; height: 26px !important; }
    #kjeldchat-header .kc-title { font-size: 18px !important; }
    #kjeldchat-input { padding: 10px 12px 12px !important; }
    #kjeldchat-input textarea { padding: 12px 14px !important; }
    /* Keep the send button square with the field at its smaller mobile height. */
    #kjeldchat-input textarea { min-height: 48px !important; }
    #kjeldchat-input button[class*="submit"], #kjeldchat-input .submit-button,
    #kjeldchat-input button {
        width: 48px !important; height: 48px !important; flex: 0 0 48px !important;
        min-height: 48px !important; margin-left: 8px !important;
    }
}

/* ---- header ------------------------------------------------------------- */
/* Asymmetric padding on purpose: Gradio's column leaves ~11px above this strip, so
   centring the brand within the strip itself sits it ~5px below the midpoint between
   the card's top edge and the divider, which is the gap the eye actually reads. */
#kjeldchat-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 13px 22px 23px; border-bottom: 1px solid rgba(255,255,255,.06);
}
#kjeldchat-header .kc-brand { display: flex; align-items: center; gap: 12px; }
#kjeldchat-header .kc-avatar {
    width: 44px; height: 44px; border-radius: 50%; flex: 0 0 44px;
    background: linear-gradient(135deg, var(--kc-accent), var(--kc-accent-2));
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 14px -4px rgba(124, 58, 237, .8);
}
#kjeldchat-header .kc-avatar svg { width: 34px; height: 34px; }
#kjeldchat-header .kc-title { font-size: 22px; font-weight: 650; color: #f2f4fb; letter-spacing: .2px; }
#kjeldchat-header .kc-badge {
    font-size: 12px; font-weight: 600; color: #c4b5fd; padding: 3px 9px;
    border: 1px solid rgba(167, 139, 250, .45); border-radius: 7px;
    background: rgba(124, 58, 237, .12);
}
/* Retry is the only control kept. The Chatbot's own buttons are switched off at source
   (buttons=[]), but Clear and Undo come from ChatInterface rather than that list, so
   they are hidden here -- by accessible label, which is semantic and stable, rather
   than by nth-child position within the group. */
#kjeldchat-wrap .icon-button-wrapper.top-panel { display: none !important; }
#kjeldchat-wrap button[aria-label="Undo"],
#kjeldchat-wrap button[aria-label="Copy message"],
#kjeldchat-wrap button[aria-label="Clear"] { display: none !important; }

/* ---- empty-state intro --------------------------------------------------- */
/* Keyed on Gradio's own .placeholder class rather than an id or inline styles in
   PLACEHOLDER_HTML, both of which Gradio's sanitiser strips. The colour is set on every
   descendant because span.md colours its children directly, which beats inheritance. */
#kjeldchat-wrap .placeholder, #kjeldchat-wrap .placeholder * {
    color: #ffffff !important;
}
#kjeldchat-wrap .placeholder .md > div {
    text-align: center !important; line-height: 1.7 !important;
}
#kjeldchat-wrap .placeholder .md > div > div:first-child {
    font-size: 17px !important; margin-bottom: 6px !important;
}
#kjeldchat-wrap .placeholder .md > div > div:last-child {
    font-size: 13.5px !important;
}

/* ---- message bubbles ---------------------------------------------------- */
/* The transcript takes whatever height is left once the header and composer have taken
   theirs, rather than a fixed 62vh that ignored the window. min-height:0 on both this
   and the column above is what actually lets a flex child shrink -- without it the
   default min-height:auto keeps the box at its content height and the composer gets
   pushed off the bottom of the card. */
#kjeldchat-wrap > .column {
    flex: 1 1 auto !important; min-height: 0 !important;
}
#kjeldchat-box {
    flex: 1 1 auto !important; height: auto !important; min-height: 0 !important;
    border: none !important; background: transparent !important;
}
/* Vertical padding only. Horizontal padding here indents the bubble but not the row's
   other children -- the Retry button and the pending indicator sit at the row's own
   left edge -- so any left/right value shows up as a misalignment, and as a sideways
   jump when the pending indicator is replaced by the bubble. The row already carries a
   20px margin, which is the inset. */
#kjeldchat-box .message-row { padding: 4px 0 !important; }

/* User: filled violet gradient, right-aligned. */
#kjeldchat-box .user-message .message-content,
#kjeldchat-box .message.user {
    background: linear-gradient(135deg, var(--kc-accent), #9333ea) !important;
    color: #fff !important; border: none !important;
    border-radius: 18px 18px 4px 18px !important;
    box-shadow: 0 6px 18px -8px rgba(124,58,237,.9) !important;
}
/* Bot: slate panel with a violet edge marking where the answer starts. */
#kjeldchat-box .bot-message .message-content,
#kjeldchat-box .message.bot {
    background: #171b2b !important;
    border: 1px solid rgba(255,255,255,.05) !important;
    border-left: 3px solid var(--kc-accent-2) !important;
    border-radius: 4px 16px 16px 4px !important;
}
/* The bubble is one element but the text lives in nested markdown nodes that carry
   Gradio's own colour -- setting it on the bubble alone leaves the text unreadable. */
#kjeldchat-box .message.bot, #kjeldchat-box .message.bot *,
#kjeldchat-box .bot-message .message-content * {
    color: #dfe3f0 !important;
}
#kjeldchat-box .message.user, #kjeldchat-box .message.user * { color: #fff !important; }

/* Cap in px, not %. The bubble's parent is shrink-to-fit, so a percentage max-width is
   circular -- the browser resolves it against the shrunken width and the bubble ends up
   ~190px, wrapping a one-line answer onto two. Same trap collapsed the user bubble
   earlier, which is why only the bot bubble is capped at all. */
#kjeldchat-box .message.bot { max-width: 660px !important; }

#kjeldchat-box .message-bubble-border { border: none !important; }
#kjeldchat-box .avatar-container { display: none !important; }

/* Gradio's per-message icon buttons (copy, retry, undo) default to a white pill. */
#kjeldchat-box .message-buttons, #kjeldchat-box .message-buttons-left,
#kjeldchat-box .message-buttons-right {
    background: transparent !important; border: none !important;
    box-shadow: none !important;
}
#kjeldchat-box .icon-button, #kjeldchat-box button.icon-button {
    background: transparent !important; color: var(--kc-muted) !important;
    border-color: rgba(255,255,255,.08) !important;
}
#kjeldchat-box .icon-button:hover { color: #ede9fe !important; }

/* ---- composer ----------------------------------------------------------- */
/* The textarea sits inside Gradio's .block wrapper, which draws its own white card --
   the textarea styling below is invisible until that wrapper is flattened. */
#kjeldchat-input, #kjeldchat-input > div, #kjeldchat-input .wrap,
#kjeldchat-input label, #kjeldchat-input .input-container,
/* The pale frame around the composer is drawn by Gradio's group/form wrappers, which
   sit *outside* #kjeldchat-input -- styling the textbox alone never reaches them. */
form, .form, .styler, .gr-group,
#kjeldchat-wrap .row, #kjeldchat-wrap .block {
    background: transparent !important; border: none !important;
    box-shadow: none !important;
}
/* Gradio's icon buttons are transparent, but the wrapper behind them is a white pill. */
#kjeldchat-wrap .icon-button, #kjeldchat-wrap button.icon-button {
    background: transparent !important; color: var(--kc-muted) !important;
    border: none !important;
}
/* The divider pipes between grouped icons are ::after pseudo-elements on each button,
   not borders -- clearing the border alone leaves them behind. */
#kjeldchat-wrap .icon-button::after, #kjeldchat-wrap .icon-button::before {
    display: none !important; content: none !important;
}
#kjeldchat-wrap .icon-button:hover { color: #ede9fe !important; }
/* Bare icons, no pill or outline behind them. */
#kjeldchat-wrap .icon-button-wrapper {
    background: transparent !important; border: none !important;
    box-shadow: none !important;
}
/* The wrap is the positioning context the top-panel buttons are pinned to. */
#kjeldchat-wrap { position: relative !important; }
#kjeldchat-input { padding: 14px 18px 18px !important; }
/* Gradio drops a loading overlay with an 80px spinner over the textbox for ~100ms each
   time a message is submitted -- a visible flash in the composer. The chat area already
   shows a typing indicator, so the spinner says nothing new. Safe to hide wholesale:
   this wrap lives inside #kjeldchat-input, whereas the generation status pill sits in a
   separate wrap inside #kjeldchat-box. */
#kjeldchat-input .wrap { display: none !important; }
/* The textarea also reflows 56 -> 54 -> 56px across submit; pinning the height stops
   the composer twitching. */
#kjeldchat-input textarea { min-height: 56px !important; }
#kjeldchat-input textarea {
    background: #141824 !important; color: #e8ebf5 !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 14px !important; padding: 16px 18px !important;
}
/* Same grey as the empty-state's "Ask a question", so the two prompts read as one voice. */
#kjeldchat-input textarea::placeholder { color: #9aa3bd !important; opacity: 1 !important; }
#kjeldchat-input textarea:focus {
    border-color: rgba(167,139,250,.55) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,.16) !important;
}
/* Send button, to the right of the field. Centred against the textarea rather than
   sitting on its baseline, which leaves it looking dropped. */
#kjeldchat-input .input-container { align-items: center !important; }
#kjeldchat-input button[class*="submit"], #kjeldchat-input .submit-button,
#kjeldchat-input button {
    background: linear-gradient(135deg, var(--kc-accent), var(--kc-accent-2)) !important;
    border: none !important; border-radius: 14px !important; color: #fff !important;
    /* Square, matching the textarea's 56px height so the two read as one control. */
    width: 56px !important; height: 56px !important; flex: 0 0 56px !important;
    min-height: 56px !important;
    align-self: center !important; margin-left: 10px !important;
}
#kjeldchat-input button:hover { filter: brightness(1.12) !important; }

/* ---- generation status ("processing | 3.7s", tokens/s) ------------------- */
/* Only the wrapper's chrome is overridden. Do NOT set width/height/inset here: Gradio
   already positions this absolutely (so it never affected layout), and forcing the box
   to auto collapses it to 0x0 -- with its overflow:hidden that clips the pill away, so
   the status flickers into existence and is never visible. */
#kjeldchat-wrap .wrap.default.minimal, #kjeldchat-wrap .wrap.translucent {
    background: transparent !important; border: none !important;
}
#kjeldchat-wrap .progress-text, #kjeldchat-wrap .meta-text {
    background: rgba(23, 27, 43, .96) !important;
    color: #cdbcff !important;
    border: 1px solid rgba(167, 139, 250, .5) !important;
    border-radius: 999px !important;
    font-size: 11.5px !important; padding: 3px 11px !important;
    box-shadow: none !important; white-space: nowrap !important;
    pointer-events: none !important;
    /* 20px matches the message rows' margin, so the pill's right edge lines up with the
       right edge of the user's question bubbles rather than sitting flush to the panel. */
    margin-right: 20px !important;
}

/* Errors surface as a Gradio toast plus an in-panel block, neither of which follows the
   theme -- on the Space a ZeroGPU quota error rendered as a full-height white sheet over
   the transcript. Worth styling: quota and cold-start errors are the ones visitors are
   most likely to actually see. */
/* .toast-wrap is the always-present positioning container, not the toast itself -- give
   it a background or border and it draws a permanent hairline across the top. */
.toast-wrap { background: transparent !important; border: none !important; }
/* Scoped to the .error variant only. Gradio uses .toast-body with .error/.warning/.info/
   .success modifiers, so styling bare .toast-body repaints every notification as an
   error -- including ZeroGPU's own "waiting for GPU" info toast, which is routine
   queueing rather than a failure. Those keep Gradio's own colours. */
.toast-body.error, .error-content,
#kjeldchat-box .error, #kjeldchat-wrap .error {
    background: #1a1020 !important; color: #fecdd3 !important;
    border: 1px solid rgba(244, 63, 94, .4) !important;
    border-radius: 14px !important; box-shadow: none !important;
}
.toast-body.error .toast-title, .toast-body.error .toast-details,
.toast-body.error .toast-text, .toast-body.error .toast-icon,
.toast-body.error .toast-close { color: #fda4af !important; }
/* The in-panel error block is sized to the chat area and pushes the layout around. */
#kjeldchat-box .error {
    height: auto !important; max-height: 160px !important; margin: 12px !important;
}

footer { display: none !important; }

/* scrollbar styling -- WebKit (Chrome/Safari/Edge) */
#kjeldchat-box *::-webkit-scrollbar { width: 10px; height: 10px; }
#kjeldchat-box *::-webkit-scrollbar-track { background: transparent; }
#kjeldchat-box *::-webkit-scrollbar-thumb { background: #2c3350; border-radius: 6px; }
#kjeldchat-box *::-webkit-scrollbar-thumb:hover { background: var(--kc-accent); }
/* Firefox */
#kjeldchat-box * { scrollbar-width: thin; scrollbar-color: #2c3350 transparent; }

#kjeldchat-input textarea::-webkit-scrollbar { width: 0; height: 0; }
#kjeldchat-input textarea { scrollbar-width: none; }
"""

def build_respond_fn(model, tokenizer, eot_id, no_penalty_ids, retriever, device, args):
    def respond(message, history):
        context, score = None, None
        if retriever is not None:
            context, score = retriever.best_passage(message.strip())
            if score < args.min_context_score:
                context = None

        prompt = (RAG_TEMPLATE.format(context=context, question=message.strip())
                   if context is not None else QA_TEMPLATE.format(question=message.strip()))
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        partial = ""
        for chunk in stream_reply(model, idx, tokenizer, eot_id, args.length, args.temperature,
                                   args.top_k, args.repetition_penalty, no_penalty_ids,
                                   penalize_from=len(ids)):
            partial += chunk
            yield partial.lstrip()

        final = partial.strip()
        if args.debug:
            if context is not None:
                final += (f"\n\n<details><summary>\U0001F4C4 Context used "
                           f"(score {score:.2f})</summary>\n\n{context}\n\n</details>")
            elif score is not None:
                final += (f"\n\n<details><summary>⚠️ No context used "
                           f"(best score {score:.2f}, below threshold "
                           f"{args.min_context_score})</summary></details>")
        yield final

    return respond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune",
                                               "checkpoints", "kjeldchat_v6.pt"))
    parser.add_argument("--tokenizer", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "base", "data", "tokenizer", "tokenizer.json"))
    parser.add_argument("--length", type=int, default=100, help="max tokens generated per reply")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True,
                         help="--no-context for closed-book (no retrieval at all)")
    parser.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_context_score", type=float, default=2.0,
                         help="2.0 for --rerank's cross-encoder scale, 0.55 for --no-rerank")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False,
                         help="show each turn's retrieved Context (score + passage) as a "
                              "collapsible block under the answer. Same default (off) as "
                              "chat.py's --debug")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer from {args.tokenizer} ...", flush=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    no_penalty_ids = continuation_byte_token_ids(tokenizer)

    print(f"Loading checkpoint from {args.checkpoint} ...", flush=True)
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
            reranker = None
            if args.rerank:
                print("Loading cross-encoder reranker ...", flush=True)
                reranker = Reranker()
            print(f"Loading passage index from {args.index_dir} ...", flush=True)
            retriever = Retriever(args.index_dir, reranker=reranker)
            print(f"  {retriever.index.meta['num_passages']:,} passages indexed", flush=True)
        else:
            print(f"(no passage index found at {args.index_dir} -- continuing closed-book)")

    respond = build_respond_fn(model, tokenizer, eot_id, no_penalty_ids, retriever, device, args)

    with gr.Blocks(title="KjeldChat") as demo:
        with gr.Column(elem_id="kjeldchat-wrap"):
            gr.HTML(HEADER_HTML)
            chat = gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(
                    elem_id="kjeldchat-box", show_label=False,
                    # No chatbot-level buttons: Retry is the only control kept, and
                    # it comes from ChatInterface, not from this list.
                    buttons=[],
                    layout="bubble",
                    avatar_images=None,
                    placeholder=PLACEHOLDER_HTML,
                ),
                textbox=gr.Textbox(elem_id="kjeldchat-input", placeholder="Type your message...",
                                    lines=1, max_lines=6, show_label=False,
                                    # Must be set here, not on ChatInterface: gr.Textbox
                                    # defaults submit_btn to False and a supplied textbox
                                    # wins, which is why the button never appeared.
                                    submit_btn=True),
                submit_btn=True,
                fill_width=True,
            )
    demo.launch(server_port=args.port, share=args.share, theme=THEME, css=CSS)


if __name__ == "__main__":
    main()
