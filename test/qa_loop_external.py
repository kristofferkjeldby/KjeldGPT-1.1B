"""
Runs the same Q/A test suite as qa_loop.py against a live external hosted model (OpenAI
or Anthropic), closed-book -- no retrieval, no RAG context handed to the model. Same
question set and Claude-judge pipeline as qa_loop.py, so the resulting summary.json is
directly comparable to the v1-v6 runs in test/runs/.

Motivation: presenting KjeldChat's own v1-v6 progression invites the natural question
"how does this compare to other models" -- but the actual system being evaluated is
KjeldGPT's finetuning + our own RAG retrieval working together, not just the generator in
isolation. Handing external models our retrieved passage would only test "can a bigger
model use context better", collapsing the comparison onto the one part of the pipeline
we didn't build differently. What we actually want to know is how our finetuned model +
RAG stacks up against whatever mechanism -- internal knowledge, their own training data,
nothing at all -- each external model relies on when it isn't handed our retrieval. So
external models get the bare question only, exactly what a generic API call to them
would look like, and answer however they answer.

There's no parameter-matched comparison available via a hosted API, and most small/early
models from KjeldChat's own scale era have since been deprecated entirely (checked live:
Anthropic's API now only serves claude-haiku-4-5, no earlier Haiku; OpenAI still serves
gpt-3.5-turbo and legacy completion models davinci-002/babbage-002/gpt-3.5-turbo-instruct).
Rather than citing published benchmark numbers for deprecated models -- a different task,
different metric, not actually comparable on the same chart -- this only ever plots
models we can genuinely run through our own test suite live.

Run from within test/ (needs ANTHROPIC_API_KEY for the judge phase always, plus
OPENAI_API_KEY if testing --backend openai/openai_completion):
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) OPENAI_API_KEY=$(cat ~/.openai_key) \\
        python3 qa_loop_external.py --backend openai --model gpt-3.5-turbo --run_name gpt-3.5-turbo
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) \\
        python3 qa_loop_external.py --backend anthropic --model claude-haiku-4-5 --run_name claude-haiku-4-5

--backend openai and anthropic (chat models) get a short system prompt covering only
the parts of the task that aren't about context at all: answer concisely, say you don't
know rather than invent a fact, and correct false premises instead of going along with
them. No mention of Context -- there isn't any.

--backend openai_completion (davinci-002, babbage-002, gpt-3.5-turbo-instruct -- legacy
completion-style models, OpenAI's modern replacement slots for the old GPT-3-era base
models) uses KjeldChat's own no-context prompt shape (`Question: ...\\nAnswer:`, chat.py's
QA_TEMPLATE) with no system prompt, since these models -- like KjeldChat itself -- are
raw text continuations with no chat scaffolding. Needs a stop sequence, unlike the chat
backends, since a completion model has no notion of "done" and will otherwise rattle on
past the answer:
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) OPENAI_API_KEY=$(cat ~/.openai_key) \\
        python3 qa_loop_external.py --backend openai_completion --model davinci-002 --run_name davinci-002

--backend huggingface runs a local open-weight model (loaded from --model_path) instead
of a hosted API -- for genuinely parameter-matched "hobby-scale" peers, since nothing at
KjeldChat's ~1.1B scale exists via OpenAI/Anthropic's own hosted lineups (their cheapest
tiers are optimized to be cheap, not small). --hf_style selects the prompt shape: "chat"
applies the model's own chat template with SYSTEM_PROMPT (for a chat-tuned model like
TinyLlama-1.1B-Chat); "completion" uses the same no-context QA_TEMPLATE as
--backend openai_completion, with the stop sequences applied post-hoc via string search
instead of a stop-sequence API (for a raw base model like GPT-2 XL). --model_path takes
any local snapshot directory or Hub id; the weights aren't kept in this repo, so
download them wherever is convenient:
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 qa_loop_external.py \\
        --backend huggingface --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --hf_style chat --run_name tinyllama-1.1b-chat
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 qa_loop_external.py \\
        --backend huggingface --model_path gpt2-xl --hf_style completion \\
        --run_name gpt2-xl
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import questions
from qa_loop import call_judge, FALSE_PREMISE_JUDGE_PROMPT
from chat import QA_TEMPLATE

SYSTEM_PROMPT = (
    "You are answering questions for a QA benchmark. Answer concisely and factually, "
    "in 1-3 sentences. If you don't know the answer, say so -- don't invent a fact. If "
    "the question embeds a false premise (something that didn't happen, or misattributes "
    "a real fact to the wrong person, place, or time), point out the error rather than "
    "answering as if it were true."
)

# qa_loop.py's own classify() assumes "no context used" is the rare exception for our
# system (retrieval came up empty), so its NO_CONTEXT_JUDGE_PROMPT only checks whether
# Wikipedia *should* cover the question -- it never judges whether the closed-book answer
# was actually correct. Here, every question is closed-book by design, so that judge would
# tag nearly every factual answer "rag_recall_failure" regardless of whether the model got
# it right. This judges actual correctness instead.
CORRECTNESS_JUDGE_PROMPT = """You are grading a model's answer to a factual question, closed-book (no retrieved context was given -- the model answered from its own training).

Question: {question}

Model's answer: {answer}

Judge whether the answer is factually correct, based on your own knowledge. Respond with ONLY a JSON object: {{"correct": true/false, "explanation": "one short sentence"}}"""


def classify_closed_book(client, model, record):
    if record["category"] == "non_factual":
        record["classification"] = "non_factual"
        return record

    if record["category"] == "false_premise":
        judgment = call_judge(client, model, FALSE_PREMISE_JUDGE_PROMPT.format(
            question=record["question"], answer=record["answer"]))
        if judgment is None:
            record["classification"] = "judge_error"
        elif judgment.get("premise_corrected"):
            record["classification"] = "premise_corrected"
        else:
            record["classification"] = "premise_accepted_hallucination"
        record["judgment"] = judgment
        return record

    judgment = call_judge(client, model, CORRECTNESS_JUDGE_PROMPT.format(
        question=record["question"], answer=record["answer"]))
    if judgment is None:
        record["classification"] = "judge_error"
    elif judgment.get("correct"):
        record["classification"] = "success"
    else:
        record["classification"] = "incorrect"
    record["judgment"] = judgment
    return record


def call_openai(client, model, question, max_tokens):
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0.8,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()


def call_anthropic(client, model, question, max_tokens):
    response = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=1.0, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def call_openai_completion(client, model, question, max_tokens):
    # No system prompt, no context -- KjeldChat's own closed-book template exactly as
    # chat.py builds it for the local model's --no-context / below-threshold path.
    prompt = QA_TEMPLATE.format(question=question)
    response = client.completions.create(
        model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.8,
        stop=["\n\n", "\nQuestion:", "\nContext:"],
    )
    return response.choices[0].text.strip()


STOP_MARKERS = ["\n\n", "\nQuestion:", "\nContext:"]


def _cut_at_stop_markers(text):
    # Local generate() has no stop-sequence API -- trim post-hoc, same markers as the
    # openai_completion backend's `stop` list.
    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def load_hf_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def call_hf_chat(model, tokenizer, device, question, max_tokens):
    import torch
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.8,
                              pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()


def call_hf_completion(model, tokenizer, device, question, max_tokens):
    import torch
    prompt = QA_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.8,
                              pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _cut_at_stop_markers(reply)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "anthropic", "openai_completion", "huggingface"], required=True)
    parser.add_argument("--model", type=str, default=None,
                         help="e.g. gpt-3.5-turbo (openai) or claude-haiku-4-5 (anthropic) -- "
                              "not used for --backend huggingface, use --model_path instead")
    parser.add_argument("--model_path", type=str, default=None,
                         help="local snapshot directory or Hub id to load for --backend "
                              "huggingface (e.g. gpt2-xl)")
    parser.add_argument("--hf_style", choices=["chat", "completion"], default=None,
                         help="prompt shape for --backend huggingface -- 'chat' applies the "
                              "model's own chat template, 'completion' uses KjeldChat's own "
                              "no-context QA_TEMPLATE for a raw base model")
    parser.add_argument("--num_factual", type=int, default=100)
    parser.add_argument("--num_factual_complex", type=int, default=146)
    parser.add_argument("--num_factual_open", type=int, default=100)
    parser.add_argument("--num_false_premise", type=int, default=50)
    parser.add_argument("--num_non_factual", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5")
    parser.add_argument("--judge_concurrency", type=int, default=5)
    parser.add_argument("--gen_concurrency", type=int, default=8,
                         help="external API calls are I/O-bound, unlike qa_loop.py's "
                              "local model which can only run one forward pass at a "
                              "time -- parallelize generation here too")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    model_label = args.model or (args.model_path and os.path.basename(args.model_path.rstrip("/")))
    run_name = args.run_name or f"{args.backend}_{model_label}"
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, f"{run_name}.jsonl")
    summary_path = os.path.join(runs_dir, f"{run_name}_summary.json")

    local_model = args.backend == "huggingface"
    if args.backend == "openai":
        from openai import OpenAI
        gen_client = OpenAI()
        call_fn = lambda question: call_openai(gen_client, args.model, question, args.max_tokens)
    elif args.backend == "openai_completion":
        from openai import OpenAI
        gen_client = OpenAI()
        call_fn = lambda question: call_openai_completion(gen_client, args.model, question, args.max_tokens)
    elif args.backend == "huggingface":
        if not args.model_path or not args.hf_style:
            raise SystemExit("--backend huggingface requires --model_path and --hf_style")
        print(f"loading {args.model_path} ...", flush=True)
        hf_model, hf_tokenizer, hf_device = load_hf_model(args.model_path)
        print(f"loaded on {hf_device}", flush=True)
        if args.hf_style == "chat":
            call_fn = lambda question: call_hf_chat(hf_model, hf_tokenizer, hf_device, question, args.max_tokens)
        else:
            call_fn = lambda question: call_hf_completion(hf_model, hf_tokenizer, hf_device, question, args.max_tokens)
    else:
        gen_client = anthropic.Anthropic()
        call_fn = lambda question: call_anthropic(gen_client, args.model, question, args.max_tokens)

    qs = ([("factual", q) for q in questions.FACTUAL[:args.num_factual]]
          + [("factual", q) for q in questions.FACTUAL_COMPLEX[:args.num_factual_complex]]
          + [("factual", q) for q in questions.FACTUAL_OPEN[:args.num_factual_open]]
          + [("false_premise", q) for q in questions.FALSE_PREMISE[:args.num_false_premise]]
          + [("non_factual", q) for q in questions.NON_FACTUAL[:args.num_non_factual]])

    def generate(item):
        category, q = item
        reply = call_fn(q)
        return {
            "category": category, "question": q, "score": None,
            "context_used": False, "context": None,
            "answer": reply.strip(),
        }

    print(f"generating {len(qs)} answers via {args.backend}:{model_label} (closed-book, no RAG context) ...", flush=True)
    if local_model:
        # One loaded model instance, one forward pass at a time -- same reasoning as
        # qa_loop.py's local KjeldChat generation, unlike the threaded API backends above.
        records = []
        for i, item in enumerate(qs):
            records.append(generate(item))
            print(f"[gen {i+1}/{len(qs)}] {item[1]}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.gen_concurrency) as executor:
            records = list(executor.map(generate, qs))

    num_to_judge = sum(1 for r in records if r["category"] != "non_factual")
    print(f"\njudging {num_to_judge} factual + false-premise answers ...", flush=True)
    judge_client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as executor:
        records = list(executor.map(lambda r: classify_closed_book(judge_client, args.judge_model, r), records))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    tally = Counter(r["classification"] for r in records if "classification" in r)
    summary = {
        "run_name": run_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": f"external:{args.backend}:{model_label} (closed-book, no RAG)",
        "num_factual": args.num_factual,
        "num_factual_complex": args.num_factual_complex,
        "num_factual_open": args.num_factual_open,
        "num_false_premise": args.num_false_premise,
        "num_non_factual": args.num_non_factual,
        "tally": dict(tally),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nwrote {len(records)} records to {out_path}")
    print(f"wrote summary to {summary_path}")
    print("\ntally:")
    for cat, count in tally.most_common():
        print(f"  {cat:<32} {count}")


if __name__ == "__main__":
    main()
