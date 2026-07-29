"""
Query-time retrieval over the passage index built by embed_passages.py. Given a
question, embeds it with the same model used to build the index (all-MiniLM-L6-v2)
and finds the closest passage by cosine similarity, for chat.py to hand to the model
as "Context: ..." instead of relying on its own thin parametric memory.

The only passage collection this project uses: data/passage_embeddings/, ~45k passages
built from Wikipedia's own Vital Articles (Level 5) list, fetched fresh via the
MediaWiki API (see fetch_vital_articles.py). This curated, much smaller pool scores far
higher on retrieval precision than a bulk full-dump chunking approach would, since it
isn't competing against millions of irrelevant passages for top-1 (see
test/qa_loop.py). Most of its recall gaps are a threshold/ranking issue rather than
missing data -- a relevant passage is usually present, just scored below
--min_context_score, or ranked below #1.

Deliberately memory-conscious even at this scale, since the same code path was written
to also work against a much larger index if this project ever needs one again:

  - embeddings.npy is opened via np.load(mmap_mode="r") -- rows are paged in by the OS
    on demand rather than loaded up front.
  - passages.txt is never read into RAM at all. A one-time pass records each line's
    starting byte offset (cached to passages.offsets.npy); looking up passage i is
    then a plain file.seek(offset[i]) + readline(), regardless of corpus size.

Search: exact brute-force cosine similarity over the full corpus (embeddings are
L2-normalized, so cosine similarity is just a dot product), done in row chunks so only
one chunk is ever cast from float16 to float32 (and resident in RAM) at a time --
measured at ~3ms/query against the current ~45k-passage corpus, negligible next to
generation time.

The passage corpus is Wikipedia-only: Gutenberg -- novels, first-person narratives,
bibliography pages -- makes unreliable Context next to a Wikipedia article, and could
out-score a better Wikipedia match purely on corpus volume, including on questions
Wikipedia answered well on its own.
"""

import json
import os

# Must be set before sentence_transformers/huggingface_hub is imported (they read it
# at import time): chat.py is meant to be self-contained, so this skips the Hub
# entirely rather than degrading to a rate-limit warning against an unauthenticated
# HF Hub request every startup. Belt-and-suspenders alongside EMBED_MODEL_PATH below,
# which loads a local snapshot rather than a Hub model name in the first place -- this
# also covers any other library code that might otherwise reach for the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "passage_embeddings")
# A full local snapshot (config + tokenizer + weights), saved once via
# SentenceTransformer('all-MiniLM-L6-v2').save(EMBED_MODEL_PATH) -- loading by this
# path rather than by Hub model name means chat.py works even if the user's HF cache
# is ever cleared, and the model ships with the repo instead of depending on it.
EMBED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")

# Bounds peak RAM for the float16->float32 cast during search to ~300MB/chunk
# (200_000 * 384 * 4 bytes), independent of total corpus size.
CHUNK_ROWS = 200_000


def build_offset_index(passages_path, offsets_path):
    """One-time scan recording each line's starting byte offset, so later lookups can
    seek() directly to passage i instead of holding passages.txt in RAM or re-scanning
    it line by line every run. Cached to disk so later runs don't re-scan passages.txt
    from scratch."""
    offsets = [0]
    with open(passages_path, "rb") as f:
        for line in f:
            offsets.append(offsets[-1] + len(line))
    offsets.pop()  # trailing entry is EOF, not a line start
    offsets = np.array(offsets, dtype=np.int64)
    np.save(offsets_path, offsets)
    return offsets


class PassageIndex:
    def __init__(self, index_dir=DEFAULT_INDEX_DIR):
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)

        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"), mmap_mode="r")

        self._passages_path = os.path.join(index_dir, "passages.txt")
        offsets_path = os.path.join(index_dir, "passages.offsets.npy")
        if os.path.exists(offsets_path):
            self.offsets = np.load(offsets_path)
        else:
            print(f"Building passage byte-offset index (one-time, "
                  f"{self.meta['num_passages']:,} lines) ...", flush=True)
            self.offsets = build_offset_index(self._passages_path, offsets_path)

    def get_passage(self, i):
        with open(self._passages_path, "rb") as f:
            f.seek(int(self.offsets[i]))
            return f.readline().decode("utf-8", errors="ignore").rstrip("\n")

    def search(self, query_vec, top_k=1):
        """query_vec: L2-normalized float32 array, shape (embedding_dim,). Returns the
        top_k (score, passage_index) pairs by cosine similarity, highest first."""
        n = self.embeddings.shape[0]
        best_scores = np.full(top_k, -np.inf, dtype=np.float32)
        best_idx = np.full(top_k, -1, dtype=np.int64)

        for start in range(0, n, CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, n)
            scores = self.embeddings[start:end].astype(np.float32) @ query_vec
            combined_scores = np.concatenate([best_scores, scores])
            combined_idx = np.concatenate([best_idx, np.arange(start, end, dtype=np.int64)])
            keep = np.argpartition(-combined_scores, top_k - 1)[:top_k]
            best_scores = combined_scores[keep]
            best_idx = combined_idx[keep]

        order = np.argsort(-best_scores)
        return [(float(best_scores[i]), int(best_idx[i])) for i in order]


class Retriever:
    """Bundles the passage index with the query embedder -- what chat.py actually
    needs: retriever.best_passage(question) -> (str, float).

    The score is handed back rather than swallowed here because "closest passage in
    the corpus" and "actually relevant" aren't the same thing -- a question like "What
    is your name?" reliably retrieves *something* about names (cosine similarity is a
    topical-overlap measure, not a relevance judgment), even though the closest match
    is usually just some unrelated person's Wikipedia biography that happens to discuss
    names, with no business being asserted as fact in answer to the question actually
    asked. Callers (see chat.py's --min_context_score) can use the score to decide
    whether the match is good enough to hand over as Context at all.

    Optional reranker (see rag_rerank.py): the bi-encoder's cosine similarity is a topical
    measure and repeatedly mis-ranks the specific right passage below a topically-
    similar wrong one, even though the right passage is often already present
    somewhere in the bi-encoder's own top-10 -- exactly what reranking that shortlist
    can recover (see test/diagnose_rag_precision.py). When a reranker
    is passed, best_passage() re-scores the top rerank_top_k candidates with it and
    returns its top pick instead of the bi-encoder's -- note the returned score is then
    the cross-encoder's own scale, not cosine similarity, so it's not comparable to a
    --min_context_score threshold tuned against the plain bi-encoder path."""

    def __init__(self, index_dir=DEFAULT_INDEX_DIR, reranker=None, rerank_top_k=10):
        # Lazy import: sentence-transformers pulls in torch/transformers, no need to
        # pay that cost for anything that only wants PassageIndex directly.
        from sentence_transformers import SentenceTransformer

        self.index = PassageIndex(index_dir)
        self.model = SentenceTransformer(EMBED_MODEL_PATH)
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    def best_passage(self, question, top_k=1):
        if self.reranker is not None:
            candidates = self.top_passages(question, top_k=self.rerank_top_k)
            return self.reranker.rerank(question, candidates)[0]
        query_vec = self.model.encode(question, normalize_embeddings=True).astype(np.float32)
        score, idx = self.index.search(query_vec, top_k=top_k)[0]
        return self.index.get_passage(idx), score

    def top_passages(self, question, top_k=10):
        """Like best_passage(), but returns every one of the top_k (passage, score)
        pairs instead of just the first -- for diagnosing whether a "wrong context"
        failure is a ranking problem (a relevant passage exists in the corpus, just
        not ranked first) or a coverage/embedding problem (nothing in the top_k is
        actually relevant)."""
        query_vec = self.model.encode(question, normalize_embeddings=True).astype(np.float32)
        return [(self.index.get_passage(idx), score) for score, idx in self.index.search(query_vec, top_k=top_k)]
