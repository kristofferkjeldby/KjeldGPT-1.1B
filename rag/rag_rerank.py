"""
Cross-encoder reranking over the bi-encoder's top-K retrieval candidates: MiniLM's
embedding-based cosine similarity search (rag_retrieve.py) is a *topical* similarity
measure -- it embeds the question and each passage independently, so it reliably
finds the right general subject but often can't distinguish the ONE specific passage
that actually answers the question from other passages about the same broader topic
(e.g. retrieving a passage about a *different* Roman emperor when asked who was the
*first* one, or John Adams when asked who was the *first* US president). A
cross-encoder reads the question and a candidate passage TOGETHER through one model,
so it can directly judge "does this specific text answer this specific question" --
much more precise, at the cost of being too slow to run over the whole corpus, so it's
only ever applied to a shortlist the cheap bi-encoder search has already narrowed down.

The bi-encoder's top-10 shortlist typically already contains the actually-relevant
passage in cases where its plain top-1 pick is wrong (see
test/diagnose_rag_precision.py) -- reranking recovers those by re-scoring the
shortlist directly against the question. It can't help cases where nothing in the
top-10 is relevant at all -- a genuine coverage/embedding gap, not a ranking problem,
and not what this is meant to fix.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~23M params). An L-12-v2 upgrade was tried
after test/runs/v7_rag_precision_failure_diagnosis.jsonl found 39/82 rag_precision_failure
cases were "lookup_issue" -- the correct passage already sat in the bi-encoder's own
top-10, L-6 just never promoted it to rank 1. rag/compare_rerankers.py's one-off
comparison found L-12 fixed 5/39 of those with zero regressions, but once the eval
judge's own bias was corrected (see test/runs/*_judge_disagreements.csv and the sanity
pass on top of them), the apparent v7->v8->v9 improvement from the reranker swap and a
lower decoding temperature turned out to be within noise -- v7's original L-6 config
scored highest of all tested variants (v4-v9, all sharing the same model weights) after
correction. Reverted back to L-6 accordingly; if this is revisited, recalibrate against
the corrected judge, not the original one. Still small enough to run fine on CPU, no
ongoing API cost. Loaded from a local snapshot (see CROSS_ENCODER_MODEL_PATH), same
offline-first reasoning as rag_retrieve.py's EMBED_MODEL_PATH: saved once via
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').save(CROSS_ENCODER_MODEL_PATH)
so chat.py keeps working even if the HF cache is ever cleared.

The cross-encoder's output is a raw relevance logit (unbounded, roughly -11 to +11), not
a cosine similarity, so the two scales are not interchangeable. chat.py's
--min_context_score default of 2.0 is calibrated for this scale; running --no-rerank
requires passing the bi-encoder-scale floor explicitly (--min_context_score 0.55).
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

CROSS_ENCODER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "models", "cross-encoder-ms-marco-MiniLM-L6-v2")


class Reranker:
    def __init__(self, model_path=CROSS_ENCODER_MODEL_PATH):
        # Lazy import: sentence-transformers pulls in torch, no need to pay that cost
        # for anything that only wants the bi-encoder retrieval path.
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_path)

    def rerank(self, question, candidates):
        """candidates: list of (passage, bi_encoder_score) pairs, e.g. from
        Retriever.top_passages() -- the bi_encoder_score is discarded here, since it's
        not on the same scale as this reranker's own score and isn't meaningful mixed
        in with it. Returns [(passage, cross_encoder_score), ...] re-sorted by the
        cross-encoder's relevance score, highest first."""
        passages = [passage for passage, _ in candidates]
        scores = self.model.predict([(question, passage) for passage in passages])
        order = sorted(range(len(passages)), key=lambda i: -scores[i])
        return [(passages[i], float(scores[i])) for i in order]
