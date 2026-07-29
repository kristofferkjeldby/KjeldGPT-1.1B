"""
Encodes every candidate context passage (same chunking as generate_qa.py's
load_wikipedia_passages, reused here so the retrieval pool is exactly the same shape --
same MAX_PASSAGE_TOKENS truncation, same single-line collapsing -- as what the model
was actually finetuned to read) into a sentence embedding, for retrieval at chat.py
inference time: embed the user's question with the same model, find the nearest
passage(s) by cosine similarity, and hand those to the model as "Context: ..." instead
of hoping it recalls the fact from its own (thin, 1.1B-parameter) memory.

Wikipedia-only: retrieval excludes Gutenberg entirely -- novels, first-person
narratives, and bibliography pages compete with, and often out-score, better Wikipedia
matches purely on corpus volume, making unreliable Context (see rag_retrieve.py's
module docstring).

Model: all-MiniLM-L6-v2 (sentence-transformers) -- small (~80MB), runs fine on CPU,
fully offline after the one-time HuggingFace download, 384-dim embeddings. Chosen for
the same reason as pyttsx3 for chat.py's speech: no ongoing API cost, no network
dependency at inference time.

Passage loading is parallelized across --workers processes (regex splitting +
per-passage tokenizer truncation is pure-Python CPU work, one process per shard file;
defaults to os.cpu_count()). Encoding auto-picks the fastest available path: CUDA if
present (single process, batched on the GPU -- this tiny model saturates a GPU in
minutes even for millions of passages), else SentenceTransformer's CPU multi-process
pool across --workers processes. Pass --device to force one or the other.

Run from within rag/. This is the command that builds the index actually in use --
45,104 Vital-article passages, a few minutes on CPU:

    cd rag
    python3 embed_passages.py --passages_file data/primary_articles/vital_passages.txt \\
        --dtype float16

Delete data/passage_embeddings/passages.offsets.npy first if passages.txt changed --
rag_retrieve.py rebuilds that byte-offset cache only when it is missing, so a stale one
silently returns text from the wrong offsets.

--wikipedia_dir switches to chunking a full clean_wikipedia dump instead (the original
~5.0M-passage approach, kept working but superseded -- see fetch_vital_articles.py).

Requires: pip install sentence-transformers
"""

import os

# Must be set before numpy/torch/tokenizers are imported anywhere in this process --
# each of the N encoding worker processes otherwise independently grabs a full BLAS/OMP
# thread pool sized to all visible cores, so N workers oversubscribe the machine by
# ~Nx and spend most of their "100% CPU" on context-switching/cache-thrashing instead
# of useful work. Since sentence-transformers spawns workers via a fresh interpreter
# (multiprocessing "spawn" context), setting os.environ here before that spawn is
# enough -- each child inherits these and initializes its native thread pools to 1.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tokenizers import Tokenizer

# generate_qa.py lives in finetune/data/ -- the Q/A corpus and the RAG passage pool
# reuse the exact same chunking (MAX_PASSAGE_CHARS/MIN_PASSAGE_CHARS/truncate_to_token_budget)
# so the retrieval pool is exactly the same shape as what the model was finetuned to
# read; this is the one place rag/ reaches into finetune/ rather than the reverse.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finetune", "data"))
from generate_qa import MAX_PASSAGE_CHARS, MIN_PASSAGE_CHARS, TOKENIZER_PATH, WIKIPEDIA_DIR, truncate_to_token_budget

MODEL_NAME = "all-MiniLM-L6-v2"


def _process_shard(args):
    """Runs in a worker process -- re-loads its own Tokenizer (tokenizers.Tokenizer
    objects aren't picklable across the process pool boundary) rather than sharing
    one, which is cheap next to the actual text processing."""
    path, tokenizer_path = args
    tokenizer = Tokenizer.from_file(tokenizer_path)
    passages = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for article in text.split("<|endoftext|>"):
        article = article.strip()
        if len(article) >= MIN_PASSAGE_CHARS:
            passages.append(truncate_to_token_budget(tokenizer, article[:MAX_PASSAGE_CHARS]))
    return passages


def load_wikipedia_passages_parallel(wikipedia_dir, tokenizer_path, workers, log):
    shard_paths = [os.path.join(wikipedia_dir, f) for f in sorted(os.listdir(wikipedia_dir))]
    passages = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_process_shard, [(p, tokenizer_path) for p in shard_paths]):
            passages.extend(result)
            done += 1
            if done % 200 == 0 or done == len(shard_paths):
                log(f"  loaded {done}/{len(shard_paths)} shards, {len(passages):,} passages so far")
    return passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_dir", type=str, default=WIKIPEDIA_DIR,
                         help="defaults to the local raw_sample -- pass the full corpus's "
                              "clean_wikipedia dir if you have it. "
                              "Ignored if --passages_file is given.")
    parser.add_argument("--passages_file", type=str, default=None,
                         help="a plain text file, one already-cleaned/truncated passage per "
                              "line (e.g. fetch_vital_articles.py's output) -- skips the "
                              "wikipedia_dir chunking entirely, for embedding a pre-built passage "
                              "list instead of raw wikipedia shards")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(os.path.dirname(__file__), "data", "passage_embeddings"))
    parser.add_argument("--batch_size", type=int, default=None,
                         help="defaults to 512 on GPU, 64 on CPU")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"],
                         help="float16 halves storage at negligible cosine-similarity "
                              "precision loss -- use it for the full-corpus run, where "
                              "size actually matters")
    parser.add_argument("--limit", type=int, default=None,
                         help="only embed the first N passages (smoke test)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                         help="CPU-path only: processes for both passage loading and "
                              "encoding -- defaults to all available cores")
    parser.add_argument("--progress_chunk", type=int, default=200_000,
                         help="log a progress/rate/ETA line every this many passages encoded")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                         help="encoding device -- defaults to cuda if available, else the "
                              "--workers-parallel CPU path. A GPU encodes this small a model "
                              "far faster than any number of CPU cores, so prefer it when present.")
    args = parser.parse_args()

    # Lazy import -- sentence-transformers pulls in torch/transformers, no need to pay
    # that import cost for anything that just wants generate_qa.py's helper functions.
    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    if args.passages_file:
        log(f"loading pre-chunked passages from {args.passages_file} ...")
        with open(args.passages_file, encoding="utf-8", errors="ignore") as f:
            passages = [line.rstrip("\n") for line in f if line.strip()]
    else:
        log(f"loading passages from {args.wikipedia_dir} ({args.workers} workers, same chunking "
            f"as generate_qa.py) ...")
        passages = load_wikipedia_passages_parallel(args.wikipedia_dir, args.tokenizer_path, args.workers, log)
    log(f"pool: {len(passages):,} passages")
    if args.limit:
        passages = passages[:args.limit]
    log(f"embedding {len(passages):,} passages this run" + (" (--limit)" if args.limit else ""))

    log(f"loading embedding model {MODEL_NAME} (device={device}) ...")
    model = SentenceTransformer(MODEL_NAME, device=device if device == "cuda" else None)

    batch_size = args.batch_size if args.batch_size is not None else (512 if device == "cuda" else 64)
    dtype = np.float16 if args.dtype == "float16" else np.float32

    def encode_chunked(encode_one_chunk):
        """Shared progress-logging loop -- encode_one_chunk(list[str]) -> np.ndarray."""
        chunks = []
        t_encode0 = time.time()
        for i in range(0, len(passages), args.progress_chunk):
            chunk = passages[i:i + args.progress_chunk]
            chunks.append(encode_one_chunk(chunk))
            done = i + len(chunk)
            elapsed = time.time() - t_encode0
            rate = done / elapsed
            eta_min = (len(passages) - done) / rate / 60
            log(f"  encoded {done:,}/{len(passages):,} passages "
                f"({rate:.1f}/s, eta {eta_min:.1f} min)")
        return np.concatenate(chunks, axis=0)

    if device == "cuda":
        log(f"encoding {len(passages):,} passages on GPU (batch_size={batch_size}) "
            f"-- single process, CUDA does the parallel work")
        embeddings = encode_chunked(lambda chunk: model.encode(
            chunk, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
        )).astype(dtype)
    elif args.workers > 1:
        log(f"encoding {len(passages):,} passages (batch_size={batch_size}, "
            f"{args.workers} CPU worker processes) -- this is the slow part")
        pool = model.start_multi_process_pool(target_devices=["cpu"] * args.workers)
        try:
            embeddings = encode_chunked(lambda chunk: model.encode(
                chunk, pool=pool, batch_size=batch_size,
                normalize_embeddings=True,  # so retrieval can use a plain dot product for cosine similarity
            )).astype(dtype)
        finally:
            model.stop_multi_process_pool(pool)
    else:
        embeddings = model.encode(
            passages, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(dtype)
    log(f"encoded: {embeddings.shape} ({args.dtype})")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "passages.txt"), "w", encoding="utf-8") as f:
        for p in passages:
            f.write(p.replace("\n", " ") + "\n")  # already single-line, but guard against surprises
    np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)

    meta = {
        "model_name": MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": args.dtype,
        "num_passages": len(passages),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"wrote {args.out_dir}/{{passages.txt,embeddings.npy,meta.json}}")


if __name__ == "__main__":
    main()
