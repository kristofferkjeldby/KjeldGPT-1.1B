"""
Same 426-question suite, RAG pipeline, and judging logic as qa_loop.py, but generating
answers from the finetuned Llama 3.2 1B checkpoint (llama/finetune_llama.py's output)
via transformers' AutoModelForCausalLM.generate() instead of this project's own
model.py/GPT class. Reuses the exact same Retriever/Reranker and Context/Question/Answer
prompt templates as KjeldChat's own "existing RAG" arm, so the two runs are apples-to-
apples: same retrieval, same questions, same judge -- only the generator differs.

Run from within test/ (needs ANTHROPIC_API_KEY for the judge phase):
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 qa_loop_llama.py --run_name llama_v1
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag_retrieve import Retriever
from rag_rerank import Reranker
import questions
from qa_loop import judge_and_write

QA_TEMPLATE = "Question: {question}\nAnswer:"
RAG_TEMPLATE = "Context: {context}\nQuestion: {question}\nAnswer:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "..", "llama", "checkpoints", "best"))
    parser.add_argument("--num_factual", type=int, default=100)
    parser.add_argument("--num_factual_complex", type=int, default=146)
    parser.add_argument("--num_factual_open", type=int, default=100)
    parser.add_argument("--num_false_premise", type=int, default=50)
    parser.add_argument("--num_non_factual", type=int, default=30)
    parser.add_argument("--index_dir", type=str, default=None)
    parser.add_argument("--rerank", action="store_true", default=True)
    parser.add_argument("--min_context_score", type=float, default=2.0)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5")
    parser.add_argument("--judge_concurrency", type=int, default=5)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, f"{run_name}.jsonl")
    summary_path = os.path.join(runs_dir, f"{run_name}_summary.json")
    raw_path = os.path.join(runs_dir, f"{run_name}_raw.jsonl")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print("loading retriever ...", flush=True)
    reranker = Reranker() if args.rerank else None
    retriever_kwargs = {"reranker": reranker} if args.index_dir is None else {"index_dir": args.index_dir, "reranker": reranker}
    retriever = Retriever(**retriever_kwargs)

    print(f"loading Llama checkpoint from {args.checkpoint} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"ready, ", flush=True)

    qs = ([("factual", q) for q in questions.FACTUAL[:args.num_factual]]
          + [("factual", q) for q in questions.FACTUAL_COMPLEX[:args.num_factual_complex]]
          + [("factual", q) for q in questions.FACTUAL_OPEN[:args.num_factual_open]]
          + [("false_premise", q) for q in questions.FALSE_PREMISE[:args.num_false_premise]]
          + [("non_factual", q) for q in questions.NON_FACTUAL[:args.num_non_factual]])

    records = []
    for i, (category, q) in enumerate(qs):
        passage, score = retriever.best_passage(q)
        context = passage if score >= args.min_context_score else None
        prompt = (RAG_TEMPLATE.format(context=context, question=q) if context is not None
                  else QA_TEMPLATE.format(question=q))

        ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        records.append({
            "category": category,
            "question": q,
            "score": score,
            "context_used": context is not None,
            "context": context,
            "answer": reply,
        })
        print(f"[gen {i+1}/{len(qs)}] {q}", flush=True)

    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(records)} raw (unjudged) answers to {raw_path}", flush=True)

    judge_and_write(records, args, run_name, out_path, summary_path)


if __name__ == "__main__":
    main()
