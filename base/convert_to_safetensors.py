"""
Converts a base_train.py checkpoint (raw torch.save pickle: {"model", "config", "iter",
"val_loss", "best_val_loss", "best_iter"}) into a Hugging Face Hub-ready folder:
model.safetensors, config.json, and a copy of tokenizer.json -- for KjeldGPT-1.1B's own
model.py architecture, not a transformers AutoModel class, so this doesn't wire up
AutoModel loading -- it just gets the artifacts into the Hub's expected shapes/names.

Drops the duplicate "head.weight" tensor before saving: when config.tied is True,
model.py's GPT.__init__ (see model.py's comment near "self.head.weight = self.tok_emb.weight")
aliases head.weight to tok_emb.weight at construction time, so the two keys share the
same underlying storage -- safetensors refuses to save aliased/shared tensors, and
there's no reason to double the file size storing the same ~309M-parameter matrix
twice. A GPTConfig(tied=True) model re-creates that alias automatically before
load_state_dict ever runs, so loading back just needs strict=False (see the loader
snippet this script prints at the end) to tolerate the one intentionally-missing key.

Run from within base/:
    cd base
    python3 convert_to_safetensors.py
"""

import dataclasses
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "kjeldgpt.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tokenizer", "tokenizer.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_export")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    config = ckpt["config"]

    if getattr(config, "tied", False):
        tok_emb = state_dict["tok_emb.weight"]
        head = state_dict["head.weight"]
        assert tok_emb.data_ptr() == head.data_ptr(), (
            "config.tied=True but tok_emb.weight/head.weight don't share storage -- "
            "check model.py's tying logic before dropping head.weight"
        )
        del state_dict["head.weight"]

    # safetensors requires contiguous tensors -- state_dicts from a resumed/optimizer-
    # stepped model are already contiguous in practice, but .contiguous() is a cheap
    # no-op safeguard against a future non-contiguous edge case.
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    save_file(state_dict, os.path.join(OUT_DIR, "model.safetensors"), metadata={
        "iter": str(ckpt["iter"]),
        "val_loss": str(ckpt["val_loss"]),
        "best_val_loss": str(ckpt["best_val_loss"]),
        "best_iter": str(ckpt["best_iter"]),
    })

    config_dict = dataclasses.asdict(config)
    config_dict["model_type"] = "kjeldgpt"
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    with open(TOKENIZER_PATH, "rb") as src, \
            open(os.path.join(OUT_DIR, "tokenizer.json"), "wb") as dst:
        dst.write(src.read())

    print(f"wrote {OUT_DIR}/model.safetensors, config.json, tokenizer.json")
    print(
        "\nTo load:\n"
        "    from safetensors.torch import load_file\n"
        "    from model import GPT, GPTConfig\n"
        "    import json\n"
        "    config = GPTConfig(**{k: v for k, v in json.load(open('config.json')).items() "
        "if k != 'model_type'})\n"
        "    model = GPT(config)  # re-creates the tied head.weight/tok_emb.weight alias\n"
        "    model.load_state_dict(load_file('model.safetensors'), strict=False)  "
        "# strict=False: head.weight is intentionally absent, already tied above\n"
    )


if __name__ == "__main__":
    main()
