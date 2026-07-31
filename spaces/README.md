---
title: KjeldChat 1.1B
emoji: 💬
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Chat with a 1.1B GPT trained from scratch, grounded by RAG
models:
  - kristofferkjeldby/KjeldChat-1.1B
  - kristofferkjeldby/KjeldGPT-1.1B
datasets:
  - kristofferkjeldby/KjeldChat-1.1B-rag-index
---

# KjeldChat 1.1B

Ask a question and get an answer from a 1.1B-parameter GPT **trained from scratch** —
no pretrained base, no distillation from a larger model. Answers are grounded by
retrieval over a Wikipedia passage index rather than recalled from the model's own
(thin) parametric memory.

> **Answers are machine-generated and unreviewed — they can be wrong.** This is a hobby
> project and a demonstration of a small model, not a reference to rely on.

- **Model:** [KjeldChat 1.1B](https://huggingface.co/kristofferkjeldby/KjeldChat-1.1B) — Q/A-finetuned from [KjeldGPT 1.1B](https://huggingface.co/kristofferkjeldby/KjeldGPT-1.1B)
- **Code:** [github.com/kristofferkjeldby/KjeldGPT-1.1B](https://github.com/kristofferkjeldby/KjeldGPT-1.1B)

## What to expect

This is a hobby project, and a 1.1B model is small. Calibrate accordingly:

- **Ask factual questions.** "When was the Eiffel Tower built?" works far better than
  "write me a poem" — it was finetuned on one Q/A format and never learned instruction
  following, multi-turn conversation, or system prompts.
- **Each turn is independent.** There is no conversational memory; follow-up questions
  that depend on the previous turn won't resolve.
- **It will sometimes be wrong,** and occasionally confidently so — including ignoring
  a correct retrieved passage, or stating something plainly false when retrieval doesn't
  cover the question. Nothing it produces is reviewed before you see it. Reducing the
  error rate is exactly what the project's evaluation harness measures.
- **False premises are its least reliable trained behavior.** Correcting a wrong
  assumption instead of playing along was specifically trained, with ~4,000 targeted
  pairs, but it still catches one only about 1 in 10 times — see the model card for
  exact numbers.

Graded blackbox against six other models on 426 questions, it answers 150 correctly —
ahead of GPT-2 XL (53, and larger) and TinyLlama-1.1B-Chat (73, instruction-tuned at
the same size), behind davinci-002 (191). Full numbers on the
[model card](https://huggingface.co/kristofferkjeldby/KjeldChat-1.1B).

## How it works

The question is embedded (`all-MiniLM-L6-v2`), matched against ~45k Wikipedia Vital
Articles passages, and the top candidates are reranked by a cross-encoder. If the best
passage clears a score threshold it's prepended as `Context:` before the question;
otherwise the model answers closed-book. Generation streams token by token.

Running on ZeroGPU — first request after the Space sleeps includes a cold start.
