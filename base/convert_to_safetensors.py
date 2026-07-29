"""
Converts a base_train.py or finetune_train.py checkpoint (raw torch.save pickle:
{"model", "config", "iter", "val_loss", "best_val_loss", "best_iter"}, plus
"resumed_from" for finetunes) into a Hugging Face Hub-ready folder: model.safetensors,
config.json, and a copy of tokenizer.json -- for KjeldGPT-1.1B's own model.py
architecture, not a transformers AutoModel class, so this doesn't wire up AutoModel
loading -- it just gets the artifacts into the Hub's expected shapes/names.

Both training scripts save the same dict shape and both models share model.py's
architecture, so one exporter serves both -- the defaults below target the base
checkpoint, and --checkpoint/--out_dir point it at a finetune instead.

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
    python3 convert_to_safetensors.py                        # base -> base/hf_export
    python3 convert_to_safetensors.py \
        --checkpoint ../finetune/checkpoints/kjeldchat_v6.pt \
        --out_dir ../finetune/hf_export                      # finetune -> finetune/hf_export

The tokenizer is the same file in both cases: finetuning never retrains it, so a
finetune export copies base/data/tokenizer/tokenizer.json too.
"""

import argparse
import dataclasses
import json
import os
import sys

import torch
from safetensors.torch import save_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "kjeldgpt.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "data", "tokenizer", "tokenizer.json")
OUT_DIR = os.path.join(BASE_DIR, "hf_export")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="checkpoint .pt to export (default: the base model)")
    parser.add_argument("--out_dir", default=OUT_DIR, help="folder to write the Hub artifacts into")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH, help="tokenizer.json to copy alongside the weights")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    metadata = {
        "iter": str(ckpt["iter"]),
        "val_loss": str(ckpt["val_loss"]),
        "best_val_loss": str(ckpt["best_val_loss"]),
        "best_iter": str(ckpt["best_iter"]),
    }
    # Finetune checkpoints record which base checkpoint they resumed from; keeping it in
    # the safetensors metadata is what makes a published finetune traceable to its base.
    if ckpt.get("resumed_from"):
        metadata["resumed_from"] = str(ckpt["resumed_from"])

    save_file(state_dict, os.path.join(args.out_dir, "model.safetensors"), metadata=metadata)

    config_dict = dataclasses.asdict(config)
    config_dict["model_type"] = "kjeldgpt"
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    with open(args.tokenizer, "rb") as src, \
            open(os.path.join(args.out_dir, "tokenizer.json"), "wb") as dst:
        dst.write(src.read())

    print(f"wrote {args.out_dir}/model.safetensors, config.json, tokenizer.json")
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
