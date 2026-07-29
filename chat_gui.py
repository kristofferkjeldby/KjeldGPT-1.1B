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
      <svg width="30" height="30" viewBox="0 0 40 40" fill="none" aria-hidden="true">
        <path d="M13 10.5v19M13 20l9.5-9.5M13.8 19.2l9.7 10.3" stroke="#fff"
              stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div class="kc-title">KjeldChat</div>
    <div class="kc-badge">1.1B</div>
  </div>
  <div class="kc-actions">
    <button id="kc-theme" class="kc-icon-btn" title="Toggle light / dark">
      <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
           stroke-linecap="round"><circle cx="12" cy="12" r="4.2"/>
        <path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2M5.4 5.4l1.6 1.6
                 M17 17l1.6 1.6M18.6 5.4L17 7M7 17l-1.6 1.6"/></svg>
    </button>
    <button id="kc-clear" class="kc-icon-btn" title="Clear conversation">
      <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
           stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 7h16M9.5 7V5.2h5V7M6.5 7l.9 12.1h9.2L17.5 7"/>
        <path d="M10.4 10.6v6M13.6 10.6v6"/></svg>
    </button>
  </div>
</div>
"""

# Shown in the empty chat area before the first question. Doubles as the place to set
# expectations -- one Q/A format, no conversational memory (see chat.py's templates).
PLACEHOLDER_HTML = """
<div style="text-align:center; color:#6b7492; line-height:1.7;">
  <div style="font-size:17px; color:#9aa3bd; margin-bottom:6px;">Ask a question</div>
  <div style="font-size:13.5px;">
    Answers are grounded in retrieved Wikipedia passages.<br>
    Each question is answered independently -- there is no conversational memory.
  </div>
</div>
"""

CSS = """
html, body {
    height: 100vh !important; max-height: 100vh !important; overflow: hidden !important;
}
[class*="gradio-container"] {
    max-width: 100% !important; width: 100% !important;
    max-height: 100vh !important; overflow: hidden !important;
    --input-text-size: var(--chatbot-text-size);
    --kc-bg: #070a14;
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
    background: var(--kc-panel) !important;
    border: 1px solid var(--kc-line) !important;
    border-radius: 24px !important;
    box-shadow: 0 0 0 1px rgba(0,0,0,.3), 0 24px 60px -20px rgba(88, 28, 135, .45) !important;
    overflow: hidden !important;
}

/* ---- header ------------------------------------------------------------- */
#kjeldchat-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 22px; border-bottom: 1px solid rgba(255,255,255,.06);
}
#kjeldchat-header .kc-brand { display: flex; align-items: center; gap: 12px; }
#kjeldchat-header .kc-avatar {
    width: 44px; height: 44px; border-radius: 50%; flex: 0 0 44px;
    background: linear-gradient(135deg, var(--kc-accent), var(--kc-accent-2));
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 14px -4px rgba(124, 58, 237, .8);
}
#kjeldchat-header .kc-avatar svg { width: 30px; height: 30px; }
#kjeldchat-header .kc-title { font-size: 22px; font-weight: 650; color: #f2f4fb; letter-spacing: .2px; }
#kjeldchat-header .kc-badge {
    font-size: 12px; font-weight: 600; color: #c4b5fd; padding: 3px 9px;
    border: 1px solid rgba(167, 139, 250, .45); border-radius: 7px;
    background: rgba(124, 58, 237, .12);
}
#kjeldchat-header .kc-actions { display: flex; gap: 10px; }
#kjeldchat-header .kc-icon-btn {
    width: 40px; height: 40px; border-radius: 12px; cursor: pointer;
    background: rgba(255,255,255,.02); border: 1px solid rgba(255,255,255,.09);
    color: var(--kc-muted); display: flex; align-items: center; justify-content: center;
    transition: color .15s ease, border-color .15s ease, background .15s ease;
}
/* flex: the button is a flex container, so an svg with no basis gets shrunk to a
   slice. stroke: Gradio sets `stroke` on svg globally, and CSS beats the element's
   own stroke="currentColor" attribute -- without this the icons draw in its dark
   slate on a dark button, i.e. invisibly. */
#kjeldchat-header .kc-icon-btn svg {
    width: 19px !important; height: 19px !important; flex: 0 0 19px;
    /* Not currentColor: Gradio sets `color` on the svg element itself, so currentColor
       resolves to its dark slate rather than to the button's colour. */
    stroke: #9aa3bd !important;
}
#kjeldchat-header .kc-icon-btn:hover svg { stroke: #ede9fe !important; }
#kjeldchat-header .kc-icon-btn:hover {
    color: #ede9fe; border-color: rgba(167,139,250,.5); background: rgba(124,58,237,.15);
}

/* ---- message bubbles ---------------------------------------------------- */
#kjeldchat-box { height: 62vh !important; border: none !important; background: transparent !important; }
#kjeldchat-box .message-row { padding: 4px 18px !important; }

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

/* Only the bot bubble is capped. Capping .message generally also hits the user bubble,
   whose row is sized to its content -- a percentage there collapses it to a few words
   per line. */
#kjeldchat-box .message.bot { max-width: 74% !important; }

#kjeldchat-box .message-bubble-border { border: none !important; }
#kjeldchat-box .avatar-container { display: none !important; }

/* Gradio's per-message icon buttons (copy, retry, undo) default to a white pill. */
#kjeldchat-box .message-buttons, #kjeldchat-box .message-buttons-left,
#kjeldchat-box .message-buttons-right {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 12px !important;
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
}
#kjeldchat-wrap .icon-button:hover { color: #ede9fe !important; }
#kjeldchat-wrap .icon-button-wrapper {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 10px !important; box-shadow: none !important;
}
/* Gradio puts its own Clear in the chat panel's top-right; the header already has one,
   and two trash icons in one view invites the "which of these does what?" pause. */
#kjeldchat-wrap .icon-button-wrapper.top-panel { display: none !important; }
#kjeldchat-input { padding: 14px 18px 18px !important; }
#kjeldchat-input textarea {
    background: #141824 !important; color: #e8ebf5 !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 14px !important; padding: 16px 18px !important;
}
#kjeldchat-input textarea::placeholder { color: #6b7492 !important; }
#kjeldchat-input textarea:focus {
    border-color: rgba(167,139,250,.55) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,.16) !important;
}
#kjeldchat-input button[class*="submit"], #kjeldchat-input .submit-button {
    background: linear-gradient(135deg, var(--kc-accent), var(--kc-accent-2)) !important;
    border: none !important; border-radius: 11px !important; color: #fff !important;
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

/* The header's Clear icon is plain HTML, so it can't own a Gradio event. This is the
   real gr.Button it forwards to -- hidden here rather than with visible=False, which
   would leave nothing in the DOM for the header button to click. */
#kc-clear-proxy { display: none !important; }
"""

# html/body are outside the css= injection's reach (same reason the background-color
# fix earlier needed a workaround) -- direct JS is the only way to suppress the outer
# page's own scrollbar, letting only #kjeldchat-box (styled above) scroll. overflow:
# hidden alone did nothing: <body> is display:flex/flex-grow:1 with no fixed height,
# and <html> uses min-height:100% (not height:100%), so taller-than-viewport content
# just grows the document instead of clipping -- height must be pinned to 100vh
# alongside overflow:hidden for the outer scrollbar to actually disappear.
HIDE_OUTER_SCROLLBAR_JS = """
() => {
    // Gradio boots in light mode unless told otherwise, which leaves its own components
    // (textbox, icon buttons, message text) white-on-white inside the dark shell the CSS
    // paints. Adding .dark switches Gradio's own variables over; the header's toggle then
    // flips this same class.
    if (!document.documentElement.classList.contains('light-forced')) {
        document.documentElement.classList.add('dark');
        document.body.classList.add('dark');
    }

    const pin = () => {
        document.documentElement.style.overflow = 'hidden';
        document.documentElement.style.height = '100vh';
        document.body.style.overflow = 'hidden';
        document.body.style.height = '100vh';
    };
    pin();
    // Re-applied on a short delay and on any later DOM change -- the SPA's own
    // post-mount layout pass (chat history loading, font metrics settling) can
    // overwrite a same-tick style set, and this is cheap enough to just keep reasserting.
    setTimeout(pin, 300);
    new MutationObserver(pin).observe(document.body, {childList: true, subtree: true});

    // Header actions. These bind to elements this file owns (HEADER_HTML's own ids and
    // #kc-clear-proxy), never to Gradio's internal markup, so a Gradio upgrade can
    // restyle the chat area without silently breaking them.
    const wire = () => {
        const theme = document.querySelector('#kc-theme');
        if (theme && !theme.dataset.bound) {
            theme.dataset.bound = '1';
            theme.addEventListener('click', () => {
                document.documentElement.classList.toggle('dark');
                document.body.classList.toggle('dark');
            });
        }
        // The visible Clear icon forwards to the hidden gr.Button, which is what
        // actually resets both the chatbot and ChatInterface's history state.
        const clear = document.querySelector('#kc-clear');
        if (clear && !clear.dataset.bound) {
            clear.dataset.bound = '1';
            clear.addEventListener('click', () => {
                const proxy = document.querySelector('#kc-clear-proxy button')
                            || document.querySelector('#kc-clear-proxy');
                if (proxy) proxy.click();
            });
        }
    };
    wire();
    new MutationObserver(wire).observe(document.body, {childList: true, subtree: true});
}
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
                    # Copy only. Gradio can also render like/dislike thumbs here, but
                    # nothing in this project consumes the flags -- they would be a
                    # control that looks functional and isn't.
                    buttons=["copy"],
                    layout="bubble",
                    avatar_images=None,
                    placeholder=PLACEHOLDER_HTML,
                ),
                textbox=gr.Textbox(elem_id="kjeldchat-input", placeholder="Type your message...",
                                    lines=1, max_lines=6, show_label=False),
                submit_btn=True,
                fill_width=True,
            )
            clear_proxy = gr.Button("Clear", elem_id="kc-clear-proxy")
            # Resets the visible transcript *and* ChatInterface's own history state --
            # clearing only the former would let old turns reappear on the next reply.
            clear_proxy.click(lambda: ([], []), outputs=[chat.chatbot, chat.chatbot_state])
    demo.launch(server_port=args.port, share=args.share, theme=THEME, css=CSS, js=HIDE_OUTER_SCROLLBAR_JS)


if __name__ == "__main__":
    main()
