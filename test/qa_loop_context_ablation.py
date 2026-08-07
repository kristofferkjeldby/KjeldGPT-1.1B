"""
RAG ablation: runs the same 426-question suite and judging pipeline as qa_loop.py, but
with --context_mode controlling where the Context handed to the model comes from,
instead of always using live retrieval:

  retrieval  -- same as qa_loop.py: rag_retrieve.Retriever + rag_rerank.Reranker,
                exactly what production chat.py does. The "existing RAG" arm.
  none       -- context is always None regardless of retrieval -- the "no RAG" arm,
                measuring how much of KjeldChat's performance retrieval buys at all.
  override   -- context comes from --overrides_path (generate_optimal_contexts.py's
                output: question -> hand-verified, single-topic, extractive passage),
                never from the retriever -- the "optimal RAG" arm, measuring the
                ceiling of the current single-passage retrieval architecture. Only
                covers the 396 factual/false-premise questions (see that script's
                docstring for why non_factual is excluded); non_factual questions still
                fall back to no-context, matching their existing treatment.

All three modes share qa_loop.py's classify()/judge_and_write() logic unchanged, so
runs across modes are directly comparable to each other and to the v1-v7/KjeldChat
track record in runs/.

Run from within test/ (needs ANTHROPIC_API_KEY; --context_mode retrieval/override load
the reranker, others don't):
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 qa_loop_context_ablation.py \\
        --context_mode none --run_name v7_no_rag
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 qa_loop_context_ablation.py \\
        --context_mode override --overrides_path optimal_contexts.jsonl --run_name v7_optimal_rag
"""
import argparse
import json
import os
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model import GPT, GPTConfig
from chat import continuation_byte_token_ids, stream_reply, RAG_TEMPLATE, QA_TEMPLATE
from rag_retrieve import Retriever
from rag_rerank import Reranker
import questions
from qa_loop import judge_and_write


def load_overrides(path):
    overrides = {}
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") in ("ok", "needs_review") and rec.get("passage"):
                overrides[rec["question"]] = rec["passage"]
            else:
                skipped += 1
    print(f"loaded {len(overrides)} override contexts from {path} ({skipped} skipped -- "
          f"failed generation/verification)", flush=True)
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_mode", type=str, required=True,
                         choices=["retrieval", "none", "override"])
    parser.add_argument("--overrides_path", type=str, default=None,
                         help="required for --context_mode override -- generate_optimal_contexts.py's output")
    parser.add_argument("--num_factual", type=int, default=100)
    parser.add_argument("--num_factual_complex", type=int, default=146)
    parser.add_argument("--num_factual_open", type=int, default=100)
    parser.add_argument("--num_false_premise", type=int, default=50)
    parser.add_argument("--num_non_factual", type=int, default=30)
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "finetune", "checkpoints", "kjeldchat_v8.pt"))
    parser.add_argument("--tokenizer", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "base", "data", "tokenizer", "tokenizer.json"))
    parser.add_argument("--index_dir", type=str, default=None)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--min_context_score", type=float, default=0.55)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5")
    parser.add_argument("--judge_concurrency", type=int, default=5)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    if args.context_mode == "override" and not args.overrides_path:
        raise SystemExit("--context_mode override requires --overrides_path")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, f"{run_name}.jsonl")
    summary_path = os.path.join(runs_dir, f"{run_name}_summary.json")
    raw_path = os.path.join(runs_dir, f"{run_name}_raw.jsonl")

    overrides = load_overrides(args.overrides_path) if args.context_mode == "override" else None

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    no_penalty_ids = continuation_byte_token_ids(tokenizer)

    retriever = None
    if args.context_mode == "retrieval":
        print("loading retriever ...", flush=True)
        reranker = Reranker() if args.rerank else None
        retriever_kwargs = {"reranker": reranker} if args.index_dir is None else {"index_dir": args.index_dir, "reranker": reranker}
        retriever = Retriever(**retriever_kwargs)

    print(f"loading model on {device} ...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(vocab_size=50257, block_size=1024, n_layer=36, n_head=24, n_embd=1536,
                        dropout=0.0, tied=True)
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"ready in {time.time()-t0:.1f}s", flush=True)

    qs = ([("factual", q) for q in questions.FACTUAL[:args.num_factual]]
          + [("factual", q) for q in questions.FACTUAL_COMPLEX[:args.num_factual_complex]]
          + [("factual", q) for q in questions.FACTUAL_OPEN[:args.num_factual_open]]
          + [("false_premise", q) for q in questions.FALSE_PREMISE[:args.num_false_premise]]
          + [("non_factual", q) for q in questions.NON_FACTUAL[:args.num_non_factual]])

    records = []
    missing_overrides = 0
    for i, (category, q) in enumerate(qs):
        if args.context_mode == "retrieval":
            passage, score = retriever.best_passage(q)
            context = passage if score >= args.min_context_score else None
        elif args.context_mode == "none":
            context, score = None, None
        else:  # override
            context = overrides.get(q)
            score = None
            if context is None and category != "non_factual":
                missing_overrides += 1

        if context is not None:
            prompt = RAG_TEMPLATE.format(context=context, question=q)
        else:
            prompt = QA_TEMPLATE.format(question=q)
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        reply = "".join(chunk for chunk in stream_reply(
            model, idx, tokenizer, eot_id, args.length, args.temperature, args.top_k,
            args.repetition_penalty, no_penalty_ids, penalize_from=len(ids)))

        records.append({
            "category": category,
            "question": q,
            "score": score,
            "context_used": context is not None,
            "context": context,
            "answer": reply.strip(),
        })
        print(f"[gen {i+1}/{len(qs)}] {q}", flush=True)

    if missing_overrides:
        print(f"\nWARNING: {missing_overrides} factual/false-premise questions had no "
              f"override context (fell back to no-context) -- check {args.overrides_path} "
              f"for failed generations", flush=True)

    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(records)} raw (unjudged) answers to {raw_path}", flush=True)

    judge_and_write(records, args, run_name, out_path, summary_path)


if __name__ == "__main__":
    main()
