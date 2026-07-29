---
license: cc-by-sa-4.0
language:
  - en
task_categories:
  - question-answering
  - feature-extraction
tags:
  - rag
  - retrieval
  - wikipedia
  - embeddings
---

# KjeldChat 1.1B — RAG passage index

The retrieval half of [KjeldChat 1.1B](https://huggingface.co/kristofferkjeldby/KjeldChat-1.1B):
a prebuilt passage index over Wikipedia, so the model can answer from a retrieved
passage instead of from a 1.1B model's thin parametric memory.

Published so the [Space](https://huggingface.co/spaces/kristofferkjeldby/KjeldChat-1.1B)
can load it at startup, and so anyone running the model locally doesn't have to spend
hours refetching and re-embedding it.

## Contents

**45,104 passages**, embedded with `all-MiniLM-L6-v2`.

| File | Size | What it is |
|---|---|---|
| `embeddings.npy` | 34MB | float16 passage embeddings, `(45104, 384)` |
| `passages.txt` | 50MB | the passage text itself, newline-delimited |
| `passages.offsets.npy` | 361KB | byte offsets into `passages.txt`, for seeking without loading it all |
| `meta.json` | <1KB | passage count and index metadata |

## Source

Passages come from Wikipedia's own community-curated
[Vital Articles (Level 5, ~45k titles)](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles)
list, fetched as live article text through the MediaWiki API. This is real Wikipedia
text — not LLM-generated, which is a deliberate constraint of the parent project.

The curated list was chosen over chunking a full Wikipedia dump after a head-to-head
comparison on a 426-question suite: the much smaller, curated pool scored substantially
higher on retrieval precision, because a correct passage isn't competing for top-1
against millions of irrelevant ones.

Wikipedia-only, deliberately. The base model's pretraining corpus also includes
Project Gutenberg, but Gutenberg passages — novels, first-person narrative,
bibliography pages — made unreliable context and could outrank a better Wikipedia match
on sheer volume.

## Usage

Embeddings are `all-MiniLM-L6-v2`, so queries must be embedded with the same model.
Retrieval and cross-encoder reranking code:
[rag/rag_retrieve.py](https://github.com/kristofferkjeldby/KjeldGPT-1.1B/blob/main/rag/rag_retrieve.py)
and [rag/rag_rerank.py](https://github.com/kristofferkjeldby/KjeldGPT-1.1B/blob/main/rag/rag_rerank.py).

```python
from huggingface_hub import hf_hub_download

for name in ("embeddings.npy", "meta.json", "passages.txt", "passages.offsets.npy"):
    hf_hub_download("kristofferkjeldby/KjeldChat-1.1B-rag-index", name,
                    repo_type="dataset", local_dir="rag/data/passage_embeddings")
```

That path is `rag_retrieve.DEFAULT_INDEX_DIR`, so `Retriever()` then picks it up with no
further configuration.

## Rebuilding it

Fully reproducible from the parent repo — `rag/fetch_vital_articles.py` refetches the
article text from the tracked ~45k-title list, and `rag/embed_passages.py` re-embeds it.

## License

Wikipedia text is CC BY-SA 4.0, and this index is a derivative of it, so the same
licence applies. The code that built it is MIT.
