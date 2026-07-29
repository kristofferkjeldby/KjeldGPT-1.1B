# KjeldChat 1.1B — Q/A finetuning run (from KjeldGPT 1.1B)

Continues the base run described in `../base/BASE_PARAMS.md` on a synthetic Q/A corpus,
teaching the model to answer questions in a fixed format rather than just continue
prose, grounded by retrieval-augmented generation (RAG) over a passage index built
from the same corpus (see "Retrieval (RAG)" below).

**Round 4, targeted corpora + passage quality (current):** kept Round 3's recipe
(dropout 0.2, peak_lr 1e-5) fixed and changed only the data, one variable at a time,
measured each time by `test/qa_loop.py` against a 426-question suite. Three additions,
in order: a **grounding** corpus aimed at demonstrated context-discarding failures and
a **false-premise** pair of corpora that teach the model to correct a question's wrong
assumption instead of playing along (both below under "Corpus"), then two **passage
quality** fixes to the Context text itself -- clipping every passage back to a complete
sentence, and stripping IPA pronunciation guides. See "Round 4 results" below for what
each bought. The current production checkpoint is that arc's last run, `v6`.

**Round 3, corpus scale-up + regularization tuning:** the two-stage round that
established the recipe Round 4 holds fixed.

Round 2 (12,124 pairs, Wikipedia-only -- see "Wikipedia-only" below) produced Round 3's
starting point, but real usage surfaced a
consistent failure mode even on questions that retrieved a genuinely relevant,
high-scoring Context: e.g. "What is the birthplace of Mozart?" retrieved the
"Mozart's birthplace" Wikipedia article at score 0.691 (correctly stating Salzburg,
Austria), yet the model's answer ignored that fact and invented an unrelated,
irrelevant detail instead. Checking Round 2's own training log ruled out "just needs
more steps": the run legitimately early-stopped from *overfitting* at step 560/961
(val loss rising for 5 consecutive evals while train loss kept falling), well short of
its planned budget -- on a corpus this size, the model ran out of room to generalize
the "trust the Context" behavior before it could fully consolidate. **Stage 3a** tripled
the corpus (see "Corpus scale-up" below) rather than changing the recipe, and resumed
from the base run's later checkpoint (step 542,868 at transfer time, up from Round 2's
~510k) rather than continuing Round 2's finetuned weights, for the same reasons Round 2
gave for not continuing Round 1's.

Stage 3a helped (early-stopped at step 1130/2892, roughly double Round 2's absolute
step count) but didn't fully fix the failure mode -- real testing still turned up cases
like "When was the US constitution written?" (context clearly says September 17, 1787;
answer said 1791, the Bill of Rights' ratification year, a classic mixup) and the
original Mozart case recurring. **Stage 3b** (see "Regularization tuning" below) reran
finetuning from the *same base checkpoint and corpus* -- no new data -- with dropout
raised 0.1→0.2 and peak_lr lowered 2e-5→1e-5, testing the theory that overfitting speed,
not corpus size, was still the bottleneck. It early-stopped at step 1920 (best step
1870), substantially further than stage 3a, and while its best val loss was slightly
*higher* in aggregate (1.0415 vs. 3a's 0.9975), real testing on
the known failure cases showed clearly better context-grounding fidelity -- the
qualitative behavior this whole round targets isn't fully captured by aggregate val
loss. Stage 3b's recipe is the one Round 4 adopted and held fixed.

**Round 2, Wikipedia-only:** Round 1 (10,026 pairs, 2/3 Wikipedia + 1/3 Gutenberg,
resumed from a base-model snapshot at step ~314k) produced Round 2's starting point;
Round 2 replaced that corpus and index entirely rather than extending them (see
"Wikipedia-only" below for why), resuming from the base model's then-current checkpoint
rather than continuing Round 1's weights, to avoid compounding a second LR warmup/decay
onto an already-converged model and to drop Round 1's Gutenberg-grounded third of the
corpus cleanly rather than trying to un-teach it.

## Corpus scale-up (Round 3)

Triples both halves of Round 2's corpus (~30,066 context pairs, ~6,306 no-context
pairs -- targets, since generation runs to an approximate `--num_pairs` floor, not an
exact count) by combining Round 2's original pairs unchanged with freshly-generated
ones from `generate_qa.py`/`generate_no_context_qa.py` -- same prompt template, same
local Wikipedia passage pool, so the two halves are format- and quality-identical, just
drawn from different passages (passage order comes from `os.listdir`, which isn't
guaranteed stable between runs, so a fresh invocation naturally samples a
largely-different subset of the ~99k-passage pool without needing a different
`--seed`). Chosen over discarding Round 2's corpus and regenerating 3x fresh, since the
goal -- more varied grounded examples for the model to generalize the "trust the
Context" behavior from -- doesn't require throwing away already-good, already-paid-for
pairs.

## Wikipedia-only

Evaluated 200 real questions (100 factual, 100 non-factual/personal) against both the
full Gutenberg+Wikipedia passage index and a Wikipedia-only one. Findings:

- Of the factual questions where the mixed index used *any* retrieved Context, 70%
  (49/70) pulled from Gutenberg rather than Wikipedia -- Gutenberg's sheer volume
  (16.6M vs. 5.0M passages) frequently out-competed a better-suited Wikipedia match on
  retrieval score alone, since cosine similarity measures topical overlap, not
  relevance or reliability.
- Zero cases where the Wikipedia-only index failed to find a match the mixed index
  found via Wikipedia -- restricting to Wikipedia cost nothing on cases mixed already
  got right, and fixed 35/100 cases where mixed had used an inferior Gutenberg passage
  or skipped a question Wikipedia could actually answer (e.g. a Wikipedia "Moon
  landing" article scored 0.558 by exact search but was buried under closer-scoring
  Gutenberg competition in the mixed corpus).
- On non-factual questions (the ones that should get no Context at all --
  "What is your name?", "How do you feel today?"), the mixed index still injected
  *some* retrieved passage as Context 7% of the time, 6 of those 7 pulled from
  Gutenberg first-person narratives (the original motivating failure: a WPA-era oral
  history about a name change got asserted as the model's own identity). The
  Wikipedia-only index dropped that false-positive rate to 1%.

Consequence: `generate_qa.py`, `embed_passages.py`, and the RAG index are all
Wikipedia-only now. The base model's own pretraining corpus is unaffected -- see
`../base/BASE_PARAMS.md`'s note.

## Corpus

Five files, generated separately, concatenated, then shuffled (`shuffle_finetune.py`,
seed 0) and tokenized. Every one uses the same
`Context: ...\nQuestion: ...\nAnswer: ...\n<|endoftext|>\n` block format -- the
differences are in what goes in the fields, not the shape:

| File | Pairs | Generator | What it teaches |
|---|---|---|---|
| `finetune_corpus_context.txt` | 30,121 | `generate_qa.py` | answer from the given Context |
| `finetune_corpus_no_context.txt` | 6,371 | `generate_no_context_qa.py` | answer without one (`Context: N/A`, see below) |
| `finetune_corpus_context_grounding.txt` | 5,014 | `generate_grounding_qa.py` | defer to Context strictly, no embellishment |
| `finetune_corpus_context_false_premise.txt` | 3,014 | `generate_false_premise_qa.py` | correct a wrong assumption using the Context |
| `finetune_corpus_no_context_false_premise.txt` | 1,008 | `generate_false_premise_no_context_qa.py` | correct one closed-book |
| **Total** | **45,528** | | |

Tokenized (`meta.json`): 9,089,564 train tokens / 1,004,883 val tokens, 40,976 / 4,552
pairs. The 10% val split lands on pair boundaries, so no pair's tokens straddle it, and
pairs are shuffled before splitting so val selection is random rather than
whichever corpus happened to be concatenated last.

Round 2, for scale: 12,124 pairs across just the first two files.

The three later corpora are all deliberately small next to the main context half. Each
targets a specific measured failure rather than adding general volume -- the grounding
corpus was generated *from* `qa_loop.py`'s own recorded failures, so it is a direct
response to what testing found, not a guess.

### Context: N/A

Round 1's corpus was 100% context-grounded -- the model had never once seen a
`Question: ...\nAnswer:` prompt without a preceding `Context:` line, even though
`chat.py` has always had a closed-book fallback (originally for `--no-context`, now
also for when retrieval's score falls below `--min_context_score`) that hands it
exactly that out-of-distribution shape. The no-context half of this corpus fixes that:
every example uses the *same* `Context: ...` template, just with the value fixed to
the literal string `N/A` instead of a real passage, rather than a differently-shaped
prompt for the no-context case. A model conditions far more reliably on "this token's
value" than on "notice a whole line is missing" -- and `chat.py` emits the identical
`Context: N/A` sentinel at inference time, so training and inference formats match
exactly.

The no-context pairs are deliberately non-factual (identity, opinion, hypothetical,
casual chat, existential-about-being-an-AI -- see `generate_no_context_qa.py`'s
`CATEGORIES`) and deliberately *not* just factual questions with Context stripped out
-- that would teach the model to hallucinate specific facts with no signal it's doing
so, the opposite of the goal. Kept far smaller than the context half (2,102 vs.
10,022) and explicitly deduplicated (`generate_no_context_qa.py` drops any repeated
question within the batch) so the model generalizes the *mode switch* rather than
memorizing a specific answer to a specific recurring question -- notably, the model
currently improvises a different invented name every time it's asked "What is your
name?" with no context, which is worth preserving rather than training into one fixed
persona.

Generation ran concurrently (`--concurrency 8-10`, bounded `ThreadPoolExecutor` +
retry-with-backoff on rate limits) since it's I/O-bound waiting on the API.

## Tokenizer

Same tokenizer as the base run -- `../data/tokenizer/tokenizer.json`,
`vocab_size=50,257`. Token ids must match for the finetuned model to mean anything
loaded against the base checkpoint's embeddings.

## Prompt/answer masking

`tokenize_finetune.py` tokenizes each pair individually and records which tokens belong to the
`Context: ...\nQuestion: ...\nAnswer:` prefix vs. the answer (+ trailing EOT), writing a
parallel mask array (`train_mask.bin`/`val_mask.bin`) alongside the token ids.
`finetune_train.py` sets the target to `-100` at every masked (prompt) position;
`model.py`'s `F.cross_entropy(..., ignore_index=-100)` skips those positions entirely,
so the loss -- and every gradient update -- only reflects producing the answer, not
reproducing the question or the Context passage.

Verified on every corpus so far, most recently 0/45,528 pairs with a tokenizer boundary
mismatch at the `Answer:` split point and 0 dropped for exceeding `MAX_EXAMPLE_TOKENS`
(900) -- `tokenize_finetune.py` reports both counts into `meta.json` on every run.

## Model architecture

Identical to the base run and to Round 1 -- required, since we're resuming a base
checkpoint (in particular `block_size=1024`: the pretrained `pos_emb` table is fixed
at that size and can't be extended).

| Param | Value |
|---|---|
| `n_layer` | 36 |
| `n_head` | 24 |
| `n_embd` | 1536 |
| `block_size` | 1024 |
| `vocab_size` | 50,257 |
| `tied` | True |
| `dropout` | **0.2** (stage 3b, up from stage 3a/Round 2's 0.1, itself up from the base run's 0.05 -- see "Regularization tuning" below. Dropout has no learnable params, so changing it doesn't affect whether the checkpoint's weights load) |

## Batch / schedule

Same recipe as Round 2 -- unchanged by Round 3's ~3x bigger corpus (and Round 2's own
~6x jump over Round 1), since `max_iters` is already derived from the actual token
count in `meta.json` at runtime rather than hardcoded (see `--epochs` below), so a
bigger corpus just means more steps, not a different schedule or a code change.
Round 2 ran `--epochs 3.0` to max_iters=961 and early-stopped (overfitting) at step 560;
Round 3's 3x token count means the same `--epochs 3.0` default now targets max_iters
=2,892 -- more room for the "trust the Context" behavior to consolidate before hitting
the same kind of overfitting wall, which was the whole motivation for this round (see
the intro note above).

| Param | Stage 3a | Stage 3b (adopted) |
|---|---|---|
| `batch_size` | 8 (not a memory constraint, a steps constraint: a smaller batch buys more optimizer steps to actually use the LR schedule) | (unchanged) |
| `peak_lr` | 2e-5 (an order of magnitude below the base run's 2e-4 -- standard finetuning practice; the base run's peak LR against this small, narrow corpus would overwrite general capability learned over 6.6B tokens in a few hundred steps instead of adapting it) | **1e-5** -- see "Regularization tuning" below |
| `min_lr` | 2e-6 (peak/10) | **1e-6** (peak/10) |
| `warmup_iters` | 10 | (unchanged) |
| `weight_decay` | 0.1 (same as base) | (unchanged) |
| `dropout` | 0.1 | **0.2** |
| `--epochs` | 3 (default, overridable) | (unchanged) |

At `batch_size=8`, Round 2's 3 epochs over 2,626,418 tokens ≈ 963 steps; Round 3's 3
epochs over 7,898,453 tokens ≈ 2,892 steps -- still minutes, not days. Neither stage ran
to completion -- both early-stopped from overfitting well short of 2,892 (see
"Regularization tuning" below for exactly where).

## Regularization tuning (stage 3b)

Stage 3a (dropout 0.1, peak_lr 2e-5 -- same recipe as Round 2, just more data) still
showed the "context sometimes discarded" failure mode in real testing after finishing
(early-stopped at step 1130/2892). Two real examples that motivated this stage:

- "What is the birthplace of Mozart?" -- retrieved Context correctly and clearly stated
  Salzburg, Austria (score 0.691, used); the answer ignored it and invented an unrelated
  detail.
- "When was the US constitution written?" -- retrieved Context clearly stated September
  17, 1787; the answer said 1791 (the Bill of Rights' ratification year -- a real date,
  just the wrong one, suggesting the model's own strong pretrained prior was winning out
  over the weaker in-context signal).

Hypothesis: stage 3a's own training log showed it overfitting at an even *smaller*
epoch-fraction than Round 2 despite 3x the data (~1.17 epochs vs. Round 2's ~1.75, in
absolute-step terms still further, but proportionally sooner) -- suggesting overfitting
speed, not corpus size, was the active constraint on how much the "trust the Context"
behavior could consolidate before training had to stop. Stage 3b reran finetuning from
the *same base checkpoint and corpus* (no new data, no code changes to the corpus
pipeline) with `dropout` raised 0.1→0.2 and `peak_lr` lowered 2e-5→1e-5, to slow
convergence and push the overfitting wall later.

Result: stage 3b early-stopped at step 1920 (best step 1870) -- substantially further
than stage 3a's step 1130. Its best val loss (1.0415) was slightly *higher* than stage
3a's (0.9975), which on its own would read as a regression -- but real testing on both
checkpoints against the known failure cases showed stage 3b noticeably more reliable at
actually using the given Context, confirming the hypothesis: aggregate val loss reflects
general next-token prediction across the whole corpus, not specifically "does it defer
to Context over its own prior", so the two can diverge. **Stage 3b's recipe -- dropout
0.2, peak_lr 1e-5 -- is the one Round 4 adopted and held fixed across v4, v5 and v6.**

`finetune_train.py` gained a `--dropout` CLI override for this (previously hardcoded),
mirroring the existing `--peak_lr` override -- see "Batch / schedule" above for the
before/after values.

## Round 4 results

All three runs share the stage-3b recipe and the same base checkpoint; only the data
differs. Whitebox `test/qa_loop.py`, 426 questions, `--rerank --min_context_score 2.0`:

| | v4 | v5 | v6 (production) |
|---|---|---|---|
| corpus change | grounding + false-premise corpora added | + passages clipped to a complete sentence | + IPA pronunciation guides stripped |
| best val loss | 0.9975 | 0.9438 @ step 2420 | **0.9385 @ step 2700** |
| success | 105 | 122 | **124** |
| finetuning grounding failure | 122 | 98 | **97** |
| false premise accepted | 40 | 41 | **37** |
| premise corrected | 10 | 9 | **13** |

The v4→v5 step is the largest single improvement in the project, and it came from
fixing data rather than training: `generate_qa.py` had been slicing passages to a
**character** budget before tokenizing, with no word or sentence awareness, so ~98% of
passages ended mid-sentence and many ended mid-word. A pre-retraining smoke test that
merely re-fed v4's 122 grounding failures with cleanly clipped Contexts flipped 34 of
them (28%) to success on the *unmodified* v4 checkpoint -- strong enough evidence to
justify the retrain. Both fixes (`clip_to_last_sentence`, `strip_pronunciation_guide` in
`generate_qa.py`) apply to the RAG index too, since `embed_passages.py` and
`fetch_vital_articles.py` share that module.

v6's gains over v5 are individually small and within run-to-run noise. It was promoted
on the reasoning that both fixes are unambiguously correct regardless of score --
a passage with a phonetic gloss the model was never trained on is simply worse
Context -- so the bar was "does not degrade", not "must improve".

## Console / eval / checkpoint cadence

Step-based, not time-based like the base run (`base_train.py`'s 300s/1800s/3600s
assumes a multi-day run; this one finishes in minutes):

| Param | Value |
|---|---|
| `console_every` | 5 steps |
| `eval_every` | 10 steps |
| `checkpoint_every` | 20 steps |
| `eval_iters` | 20 |
| `patience` | 5 evals (code default). Round 4 ran `--patience 10` throughout -- at
  eval_every=10 the val curve is noisy enough that 5 evals sometimes stopped a run that
  was still improving. |

## Resuming / checkpoint naming

- `--resume` always points at the *base* checkpoint (`../base/checkpoints/kjeldgpt.pt`,
  now frozen at step 808,998), never at an earlier round's finetuned output. Every round
  and every Round 4 run is a fresh finetune from the same base -- see the intro's round
  notes for why. This also keeps runs comparable: v5 and v6 differ only in their corpus,
  not in what they were built on.
- Finetuning starts its own step counter and best-val tracking at 0 -- the base
  checkpoint's step/val-loss aren't comparable to this run's masked QA objective.
- Training writes two files into `--out_dir`:
  - `kjeldchat.pt` -- the *latest* weights, on the periodic `checkpoint_every` cadence
    (and at the final/early-stopped step).
  - `kjeldchat_best.pt` -- saved immediately every time val loss improves, independent
    of that cadence, so the true best-val weights are never missed between periodic
    saves. This is the one that gets kept.
- Locally, each round's `kjeldchat_best.pt` is downloaded and renamed to its run name --
  `checkpoints/kjeldchat_v4.pt`, `_v5.pt`, `_v6.pt` -- so every tested checkpoint stays
  on disk and reproducible against its `test/runs/` entry. `chat.py`, `chat_gui.py` and
  `test/qa_loop.py` name the current production one directly (`kjeldchat_v6.pt`);
  promoting a new run is a one-line change in each.

## Hardware

Same single RTX PRO 6000 Blackwell (97.9GB VRAM) as the base run -- see
`../base/BASE_PARAMS.md`'s hardware note; no part of this depends on that specific
machine. A full finetuning run is ~78 minutes, so unlike the base run this is
comfortably a sit-and-watch job rather than a multi-day one.

## Launch command

Round 4 -- every run uses exactly these flags. The recipe is held fixed so that runs
differ only by their corpus and stay comparable:

```
cd finetune
python3 finetune_train.py --dropout 0.2 --peak_lr 1e-5 --patience 10
```

Run `data/shuffle_finetune.py` and `data/tokenize_finetune.py` first. `--resume` defaults
to `../base/checkpoints/kjeldgpt.pt`; if you point it at a *copy* of that checkpoint
instead (training on a remote box, say), verify the copy by MD5 first -- a run resumed
from a different base is not comparable to the others, and nothing in the output will
reveal that.

## Retrieval (RAG)

The finetuned model answers from a Context passage handed to it at inference time
(`chat.py`), rather than only ever recalling facts from its own thin 1.1B-parameter
memory -- this is what the Context-grounded half of the corpus above actually trains
it to expect.

### Passage source

`fetch_vital_articles.py` fetches live article text via the MediaWiki API for the
~45k titles on Wikipedia's own community-curated Vital Articles (Level 5) list, giving
**45,104 passages** in `rag/data/primary_articles/vital_passages.txt`.

This replaced chunking the full local `clean_wikipedia` dump (~5.0M passages) after a
head-to-head `qa_loop.py` comparison: the curated pool scored far *higher* on retrieval
precision despite being ~100x smaller, because a correct passage no longer has to
out-score millions of irrelevant near-misses to reach top-1. The bulk-dump path still
exists as `embed_passages.py --wikipedia_dir`, but nothing uses it.

Passages are cut with `generate_qa.py`'s `truncate_to_token_budget`, so the RAG index
and the finetuning corpus are shaped identically -- the model sees the same kind of text
at inference that it saw in training, including the Round 4 clipping and
pronunciation-stripping fixes.

### Embeddings

`embed_passages.py` encodes each passage with `all-MiniLM-L6-v2` (sentence-transformers,
384-dim, L2-normalized so cosine similarity is a plain dot product), stored as float16
in `rag/data/passage_embeddings/embeddings.npy` -- 33MB at this corpus size, small
enough that `rag_retrieve.py` does an exact brute-force dot product over the whole
matrix per query in a few milliseconds. No approximate index is needed or used; the
Vital-articles corpus is what made that simplification possible.

`passages.txt` is read through a byte-offset cache (`passages.offsets.npy`) so a lookup
seeks directly to a line instead of scanning. **It is rebuilt only when missing**, so
delete it whenever `passages.txt` is rewritten -- otherwise lookups silently return
text from the wrong offsets.

### Reranking

The bi-encoder measures topical similarity, which repeatedly ranked a topically-close
wrong passage above the specifically-right one. `rag_rerank.py` takes its top-10
candidates and rescores them with a cross-encoder (`ms-marco-MiniLM-L-6-v2`), which
reads question and passage jointly rather than comparing two independent embeddings.
Enabled by default in `chat.py` (`--rerank`).

### Context threshold

Retrieval finds the closest passage by topical overlap, not relevance -- a question like
"What is your name?" reliably retrieves *something* about names, usually an unrelated
biography. `--min_context_score` discards the passage and falls back to closed-book
below that floor rather than asserting a merely topically-similar passage as fact.

The default is **2.0**, on the cross-encoder's logit scale (unbounded, roughly -11 to
+11), not the bi-encoder's 0-1 cosine scale. Passing `--no-rerank` therefore also
requires passing `--min_context_score 0.55` -- the two scales are not interchangeable,
and mixing them silently accepts or rejects everything. `chat.py`'s `debug`/`no_debug`
mid-session command shows the score and used/skipped status for every turn.
