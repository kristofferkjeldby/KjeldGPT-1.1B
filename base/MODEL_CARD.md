---
license: mit
language:
  - en
pipeline_tag: text-generation
tags:
  - gpt
  - causal-lm
  - trained-from-scratch
  - pytorch
---

# KjeldGPT 1.1B

A 1.1B-parameter GPT **trained from scratch** on a combined Gutenberg + Wikipedia
corpus. This is the *base* model: it continues text, it does not follow instructions
and it does not answer questions in any fixed format. For the Q/A-finetuned variant,
see [KjeldChat 1.1B](https://huggingface.co/kristofferkjeldby/KjeldChat-1.1B).

Source, training scripts and evaluation harness:
**https://github.com/kristofferkjeldby/KjeldGPT-1.1B**

## This is not a `transformers` model

The weights are a plain `safetensors` dump of a custom `nn.Module` defined in
[`model.py`](https://github.com/kristofferkjeldby/KjeldGPT-1.1B/blob/main/model.py).
There is no `AutoModel` mapping and `AutoModelForCausalLM.from_pretrained` will not
work. Load it with the repo's own architecture:

```python
import json
from safetensors.torch import load_file
from model import GPT, GPTConfig   # from github.com/kristofferkjeldby/KjeldGPT-1.1B

config = GPTConfig(**{k: v for k, v in json.load(open("config.json")).items()
                      if k != "model_type"})
model = GPT(config)      # re-creates the tied head.weight / tok_emb.weight alias
model.load_state_dict(load_file("model.safetensors"), strict=False)
model.eval()
```

`strict=False` is required and expected: `head.weight` is deliberately absent from the
file. With `tied=True` the head and the token embedding share one storage, and
`safetensors` refuses to serialize aliased tensors — so the ~309M-parameter matrix is
stored once, and `GPT.__init__` re-creates the alias before loading.

The tokenizer is a `tokenizers` (not `transformers`) JSON, trained jointly on this
model's own corpus:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")
```

## Architecture

| Param | Value |
|---|---|
| `n_layer` | 36 |
| `n_head` | 24 |
| `n_embd` | 1536 |
| `block_size` | 1024 |
| `vocab_size` | 50,257 |
| `tied` | True |
| `dropout` | 0.05 |
| **Total params** | **~1,098.7M** |

Decoder-only transformer, learned positional embeddings, pre-norm blocks, fused
scaled-dot-product attention. Weights are fp32.

## Training data

| Source | Size | Files |
|---|---|---|
| Gutenberg | 12.0 GB | 30,817 books |
| Wikipedia | 19.5 GB | 6,083,989 articles |
| **Combined** | **~31.5 GB** | **32,845 files** |

**7,363,682,333 tokens** total (6.63B train / 0.74B val). The model was sized from the
corpus rather than a time budget: ~3 epochs lands near the ~20-tokens-per-parameter
compute-optimal ratio.

## Training

Trained for the full 3-epoch budget, finishing at step **808,998** with a final
validation loss of **2.5010** (best 2.4886 at step 790,013) — roughly 7.5 days on a
single RTX PRO 6000 Blackwell, batch size 24, peak LR 2e-4 with cosine decay to 2e-5
after 1,000 warmup steps. Full hyperparameters and rationale:
[`base/BASE_PARAMS.md`](https://github.com/kristofferkjeldby/KjeldGPT-1.1B/blob/main/base/BASE_PARAMS.md).

## Intended use and limitations

This is a hobby project and a learning exercise, published openly in that spirit. It is
not a product and has had no safety tuning.

- **It is a text continuer.** Prompt it with the start of a passage, not a question.
- **Parametric memory is thin.** At 1.1B parameters on 7.4B tokens it will state
  confident, wrong facts. The downstream KjeldChat model addresses this with retrieval
  rather than by trusting recall — that design choice is the point of the project.
- Training data is unfiltered Gutenberg and Wikipedia text, and the model reproduces
  the biases and the dated language of both. Public-domain Gutenberg texts in
  particular skew heavily toward pre-1930 attitudes.
- English only.

## License

MIT, for the code and these weights. The underlying corpora carry their own terms
(Wikipedia: CC BY-SA; Project Gutenberg: public domain in the US, with its own license
for the collection).
