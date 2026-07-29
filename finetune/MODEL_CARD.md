---
license: mit
language:
  - en
pipeline_tag: text-generation
tags:
  - gpt
  - causal-lm
  - question-answering
  - rag
  - trained-from-scratch
  - pytorch
base_model: kristofferkjeldby/KjeldGPT-1.1B
---

# KjeldChat 1.1B

A Q/A-finetuned variant of [KjeldGPT 1.1B](https://huggingface.co/kristofferkjeldby/KjeldGPT-1.1B),
a 1.1B-parameter GPT trained from scratch. It answers questions in a fixed
`Question: ... / Answer: ...` format instead of continuing prose, and it is designed to
answer **from a retrieved passage**, not from parametric memory.

This repo holds checkpoint **v6**, the production checkpoint.

Source, RAG index builder, and evaluation harness:
**https://github.com/kristofferkjeldby/KjeldGPT-1.1B**

## This is not a `transformers` model

The weights are a plain `safetensors` dump of a custom `nn.Module` defined in
[`model.py`](https://github.com/kristofferkjeldby/KjeldGPT-1.1B/blob/main/model.py).
`AutoModelForCausalLM.from_pretrained` will not work. Load it with the repo's own
architecture:

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

`strict=False` is required and expected — `head.weight` is deliberately absent, since
it shares storage with `tok_emb.weight` under `tied=True` and `safetensors` will not
serialize aliased tensors. The alias is restored by `GPT.__init__` before loading.

Tokenizer (a `tokenizers` JSON, identical to the base model's — finetuning does not
retrain it):

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")
```

## Prompt format

The model was finetuned on exactly two templates, and it degrades badly outside them.

Closed-book:

```
Question: {question}
Answer:
```

With a retrieved passage — **this is the intended path**:

```
Context: {context}
Question: {question}
Answer:
```

Generation stops at the end of the answer. A ready-made REPL that does retrieval,
prompting and streaming for you is
[`chat.py`](https://github.com/kristofferkjeldby/KjeldGPT-1.1B/blob/main/chat.py); a
browser UI over the same pipeline is `chat_gui.py`.

## The retrieval half is not in this repo

These weights are half the system. The other half is a passage index built from
Wikipedia's community-curated
[Vital Articles (Level 5, ~45k titles)](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles),
fetched live via the MediaWiki API, embedded with `all-MiniLM-L6-v2` and reranked with
a `ms-marco-MiniLM-L-6-v2` cross-encoder. Rebuild it from the GitHub repo with
`rag/fetch_vital_articles.py` + `rag/embed_passages.py` — the ~45k-title input list is
tracked there, so the index is reproducible. Used closed-book, this model is much
weaker than the numbers below suggest.

## Finetuning

Resumed from the base checkpoint (never chained on top of a previous finetune) over a
synthetic Q/A corpus of ~45,500 pairs generated *from real Wikipedia passages*:
grounded Q/A, deliberately non-factual no-context pairs, ~4,000 false-premise
correction pairs, and a grounding corpus targeting demonstrated context-discarding
failures. Batch size 8, peak LR 1e-5 (cosine to 1e-6), dropout 0.2, early-stopped at
step **2,700** with validation loss **0.9385**.

Both the finetuning corpus and the RAG index are Wikipedia-only, even though the base
model's pretraining corpus is Gutenberg + Wikipedia: Gutenberg passages — novels,
first-person narrative, bibliography pages — made unreliable context and could outrank
a better Wikipedia match on sheer corpus volume. Full rationale:
[`finetune/FINETUNE_PARAMS.md`](https://github.com/kristofferkjeldby/KjeldGPT-1.1B/blob/main/finetune/FINETUNE_PARAMS.md).

## Evaluation

426 questions — 346 factual, 50 false-premise, 30 non-factual — run through seven
models and graded by the same judge. **Blackbox**: question in, answer out. KjeldChat's
retrieval is invisible in the comparison and no external model was handed a passage.

| Model | Correct / 426 |
|---|---|
| gpt-3.5-turbo-instruct | 331 |
| gpt-3.5-turbo | 298 |
| davinci-002 | 191 |
| **KjeldChat 1.1B** | **138** |
| babbage-002 | 94 |
| TinyLlama-1.1B-Chat | 73 |
| GPT-2 XL (1.5B) | 53 |

It clears a model larger than itself (GPT-2 XL) and an instruction-tuned model of the
same size (TinyLlama), and lands below davinci-002.

The false-premise column is the more interesting result: KjeldChat accepts a false
premise 37 times — the lowest of the five small models, against TinyLlama's 44 and
GPT-2 XL's 49 — and explicitly corrects one 13 times, against davinci-002's 8 and
GPT-2 XL's 1. That comes from ~4,000 targeted training pairs, not from scale.

## Intended use and limitations

A hobby project and a learning exercise, published openly in that spirit. Not a
product; no safety tuning.

- **Give it context.** Closed-book, a 1.1B model trained on 7.4B tokens confabulates
  confidently. Everything about this project's design assumes retrieval.
- **Stay in the prompt format.** Multi-turn conversation, system prompts, and
  instruction-following in general were never trained and do not work.
- It will still sometimes ignore a correct retrieved passage and answer from memory —
  the failure mode the grounding corpus targets and has reduced, not eliminated.
- English only. Inherits the biases of unfiltered Wikipedia and Gutenberg text.

## License

MIT, for the code and these weights. Wikipedia-derived training content is CC BY-SA.
