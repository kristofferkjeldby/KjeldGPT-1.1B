"""
Hugging Face Space entrypoint for KjeldChat 1.1B -- the same pipeline as chat_gui.py,
with the local-filesystem assumptions swapped for Hub downloads.

chat_gui.py expects a working tree: a .pt checkpoint under finetune/checkpoints/, a
passage index under rag/data/, and sentence-transformer snapshots under rag/models/.
None of that exists on a Space, so this module downloads each piece from the Hub into
the paths the existing code already looks in, then reuses chat_gui.build_respond_fn
unchanged. The UI (theme/CSS/logo) is imported from chat_gui too, so the Space and the
local GUI can't drift apart.

The one genuine behavioural difference is ZeroGPU: generation is wrapped in
@spaces.GPU so the Space borrows a GPU per request instead of holding one. Everything
below the decorator is chat_gui's code path.

Deployed by build_space.py -- see that script for how the Space repo is assembled.
Run locally with `python3 app.py` (works, but downloads ~4.7GB on first start).
"""

import os

# rag_retrieve.py and rag_rerank.py both do os.environ.setdefault("HF_HUB_OFFLINE", "1")
# at import time, to keep the local chat.py from touching the network. A Space is the
# exact inverse: every artifact comes from the Hub. setdefault means claiming the
# variable here wins, so this must run before those modules are imported (below).
os.environ["HF_HUB_OFFLINE"] = "0"

import json
import sys
import types

import gradio as gr
import spaces
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rag"))

MODEL_REPO = "kristofferkjeldby/KjeldChat-1.1B"
INDEX_REPO = "kristofferkjeldby/KjeldChat-1.1B-rag-index"

# Where the existing code expects to find things. These aren't arbitrary -- they mirror
# rag_retrieve.DEFAULT_INDEX_DIR / EMBED_MODEL_PATH and rag_rerank.CROSS_ENCODER_MODEL_PATH,
# so downloading into them means those modules need no Space-specific changes.
INDEX_DIR = os.path.join(ROOT, "rag", "data", "passage_embeddings")
EMBED_DIR = os.path.join(ROOT, "rag", "models", "all-MiniLM-L6-v2")
RERANK_DIR = os.path.join(ROOT, "rag", "models", "cross-encoder-ms-marco-MiniLM-L6-v2")

# 100 tokens at KjeldChat's size generates in a few seconds on ZeroGPU's Blackwell, but
# the budget also covers bi-encoder retrieval and cross-encoder reranking on the same
# allocation. Kept tight rather than padded: shorter declared durations get better queue
# priority for visitors.
GPU_DURATION = 60


def fetch_artifacts():
    """Pull weights, passage index and both encoders into the paths above."""
    print("Downloading model weights ...", flush=True)
    weights = hf_hub_download(MODEL_REPO, "model.safetensors")
    config_path = hf_hub_download(MODEL_REPO, "config.json")
    tokenizer_path = hf_hub_download(MODEL_REPO, "tokenizer.json")

    print("Downloading passage index ...", flush=True)
    for name in ("embeddings.npy", "meta.json", "passages.txt", "passages.offsets.npy"):
        hf_hub_download(INDEX_REPO, name, repo_type="dataset", local_dir=INDEX_DIR)

    print("Downloading retrieval encoders ...", flush=True)
    snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_dir=EMBED_DIR)
    snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2", local_dir=RERANK_DIR)

    return weights, config_path, tokenizer_path


WEIGHTS, CONFIG_PATH, TOKENIZER_PATH = fetch_artifacts()

# Imported only after fetch_artifacts(): importing these pulls in rag_retrieve, and the
# Retriever resolves its encoder path at construction time, not import time -- but the
# ordering is load-bearing for HF_HUB_OFFLINE regardless, so keep it explicit.
from model import GPT, GPTConfig
from rag_rerank import Reranker
from rag_retrieve import Retriever
from chat import continuation_byte_token_ids
from chat_gui import (CSS, HEADER_HTML, HIDE_OUTER_SCROLLBAR_JS, PLACEHOLDER_HTML, THEME,
                       build_respond_fn)

# ZeroGPU requires module-level .to("cuda") -- a CUDA emulation mode is active out here,
# and real CUDA inside @spaces.GPU. Moving the model inside the decorated function
# instead is explicitly discouraged: transfers are optimised for startup placement.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Building model ...", flush=True)
config = GPTConfig(**{k: v for k, v in json.load(open(CONFIG_PATH)).items() if k != "model_type"})
model = GPT(config)
# strict=False: head.weight is intentionally absent from the safetensors file (tied to
# tok_emb.weight, which GPT.__init__ has already re-aliased). See the model card.
model.load_state_dict(load_file(WEIGHTS), strict=False)
model.eval()
model.to(DEVICE)

print("Loading tokenizer ...", flush=True)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
eot_id = tokenizer.token_to_id("<|endoftext|>")
no_penalty_ids = continuation_byte_token_ids(tokenizer)

print("Loading retriever ...", flush=True)
retriever = Retriever(INDEX_DIR, reranker=Reranker())
print(f"  {retriever.index.meta['num_passages']:,} passages indexed", flush=True)

# build_respond_fn reads generation settings off an argparse Namespace. The Space has no
# command line, so this stands in for it -- same defaults as chat_gui.py's parser, with
# min_context_score on the reranker's cross-encoder scale (2.0), not the bi-encoder's.
args = types.SimpleNamespace(
    length=100,
    temperature=0.8,
    top_k=50,
    repetition_penalty=1.3,
    min_context_score=2.0,
    debug=False,
)

_respond = build_respond_fn(model, tokenizer, eot_id, no_penalty_ids, retriever, DEVICE, args)


@spaces.GPU(duration=GPU_DURATION)
def respond(message, history):
    """Thin ZeroGPU wrapper -- yields through chat_gui's streaming generator unchanged."""
    yield from _respond(message, history)


# Mirrors chat_gui.main()'s layout exactly -- the styling constants are imported rather
# than copied, so the Space and the local GUI stay one design.
with gr.Blocks(title="KjeldChat") as demo:
    with gr.Column(elem_id="kjeldchat-wrap"):
        gr.HTML(HEADER_HTML)
        chat = gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(
                elem_id="kjeldchat-box", show_label=False,
                buttons=[], layout="bubble", avatar_images=None,
                placeholder=PLACEHOLDER_HTML,
            ),
            textbox=gr.Textbox(elem_id="kjeldchat-input", placeholder="Type your message...",
                                lines=1, max_lines=6, show_label=False, submit_btn=True),
            submit_btn=True,
            fill_width=True,
        )

demo.launch(theme=THEME, css=CSS, js=HIDE_OUTER_SCROLLBAR_JS)
