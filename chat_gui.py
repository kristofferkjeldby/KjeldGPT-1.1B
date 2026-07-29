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

THEME = gr.themes.Origin()

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")

CSS = """
html, body {
    height: 100vh !important; max-height: 100vh !important; overflow: hidden !important;
}
[class*="gradio-container"] {
    max-width: 100% !important; width: 100% !important;
    max-height: 100vh !important; overflow: hidden !important;
    --input-text-size: var(--chatbot-text-size);
    --background-fill-primary: #222631 !important;
    background-color: #222631 !important;
}
#kjeldchat-wrap { max-width: 950px !important; margin: 0 auto !important; padding: 0 32px !important; gap: 0 !important; }
#kjeldchat-logo { padding: 0 !important; margin: 0 0 17px 0 !important; min-height: 0 !important; }
#kjeldchat-logo > div { padding: 0 !important; margin: 0 !important; border: none !important; background: none !important; min-height: 0 !important; }
#kjeldchat-logo img { max-height: 56px; width: auto; margin: 0 auto; display: block; object-fit: contain; }
#kjeldchat-logo .icon-button-wrapper { display: none !important; }
footer { display: none !important; }
#kjeldchat-box { height: 70vh !important; }

/* scrollbar styling -- WebKit (Chrome/Safari/Edge) */
#kjeldchat-box *::-webkit-scrollbar { width: 10px; height: 10px; }
#kjeldchat-box *::-webkit-scrollbar-track { background: #222631; }
#kjeldchat-box *::-webkit-scrollbar-thumb { background: #454b5c; border-radius: 6px; }
#kjeldchat-box *::-webkit-scrollbar-thumb:hover { background: #5EEAD4; }
/* Firefox */
#kjeldchat-box * { scrollbar-width: thin; scrollbar-color: #454b5c #222631; }

#kjeldchat-input textarea::-webkit-scrollbar { width: 0; height: 0; }
#kjeldchat-input textarea { scrollbar-width: none; }
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
            gr.Image(
                value=LOGO_PATH, elem_id="kjeldchat-logo",
                show_label=False, container=False, interactive=False,
                buttons=[],
            )
            gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(elem_id="kjeldchat-box", show_label=False),
                textbox=gr.Textbox(elem_id="kjeldchat-input", placeholder="Ask a question...",
                                    lines=1, max_lines=6, show_label=False),
                fill_width=True,
            )
    demo.launch(server_port=args.port, share=args.share, theme=THEME, css=CSS, js=HIDE_OUTER_SCROLLBAR_JS)


if __name__ == "__main__":
    main()
