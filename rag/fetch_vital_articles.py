"""
Fetches real, live Wikipedia article text (via the MediaWiki API's plaintext extract,
one title per request, since batching multiple titles into one request only returns
real content for one of them, silently empty-stringing the rest) for a list of titles
(Wikipedia:Vital articles), cleans and truncates each (same MAX_PASSAGE_CHARS/
token-budget truncation as finetune/data/generate_qa.py), and writes JSONL -- one
{"title": ..., "passage": ...} object per line. The title is kept alongside the passage
text so passages stay traceable to their source article and can be deduped or replaced
by title on a refetch.

The ~45k passages this produces are the project's entire RAG corpus -- chosen over
chunking the full Wikipedia dump because a curated pool scores far higher on retrieval
precision (see rag_retrieve.py's module docstring).

Uses its OWN charset filter (RAG_CHARSET_FILTER below), deliberately NOT
base/data/clean_wikipedia_dataset.py's CHARSET_FILTER -- that one is ASCII-only by
design (shared vocabulary with Gutenberg for base-model pretraining, a corpus already
built and in use, not something to retroactively change here), and strips characters
RAG passage text needs to keep: en/em dashes (e.g. a birth-death date range would
render as "1934 November 7, 2016" instead of "1934 -- November 7, 2016") and accented
characters. Since that stripping is a lossy regex substitution, there's no way to
recover those characters from already-cleaned Wikipedia text -- hence fetching fresh
here rather than reprocessing the existing corpus. RAG_CHARSET_FILTER keeps all Unicode
letters/digits (\\w) plus common prose punctuation including en/em dashes, curly
quotes, and ellipsis, rather than an ASCII-only whitelist.

Also writes the RAW (un-cleaned, un-truncated) extract to --raw_out, so a *future*
filter fix never needs another live refetch -- just reprocessing this file.

This is the only place real content enters the RAG passage database from outside the
original Wikipedia dump -- deliberately NOT Claude-generated (see the project's
Claude-role-boundary principle): every passage here is fetched, not invented.

Run from within rag/ (needs network access, no API key):
    python3 fetch_vital_articles.py --titles data/primary_articles/vital_level5_titles.txt \\
        --out data/primary_articles/vital_passages.jsonl \\
        --raw_out data/primary_articles/vital_passages_raw.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finetune", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base", "data"))
from generate_qa import MAX_PASSAGE_CHARS, MIN_PASSAGE_CHARS, TOKENIZER_PATH, truncate_to_token_budget
from tokenizers import Tokenizer

USER_AGENT = "KjeldGPT-personal-project/1.0 (contact: kristoffer@kjeldby.dk)"
API_URL = "https://en.wikipedia.org/w/api.php"

# Unicode-aware: \w keeps every script's letters/digits (accented Latin, Danish o/ae/aa,
# etc.) rather than base/data/clean_wikipedia_dataset.py's ASCII-only CHARSET_FILTER --
# see module docstring for why that one silently corrupted RAG passage text. Explicitly
# keeps en-dash/em-dash (–/—), curly quotes (‘/’/“/”), and
# ellipsis (…) alongside the plain ASCII punctuation the old filter already had, plus
# %/&/°/£/$ -- found meaningfully present (percentages, names like "White & Nerdy",
# temperatures, currency) in a 100-title smoke test of what this filter would strip.
# Everything else it drops (=, {, }, \, [, ], <, >, ^, and assorted math symbols) is
# LaTeX/math-markup residue from equation rendering leaking into the plaintext API
# extract -- already-garbled, low-value content for a Q&A passage regardless.
RAG_CHARSET_FILTER = re.compile(
    r"[^\w\s'\".,!?;:()\-–—‘’“”…/%&°£$]", re.UNICODE)


class FetchError(Exception):
    """Raised when the fetch itself kept failing (rate limit, network) -- must never
    be treated the same as a genuine "Wikipedia has no such page" response, since
    silently conflating the two would undercount real coverage as fake gaps."""


def fetch_extract(title, max_retries=8):
    """Returns the plaintext extract (str, possibly empty), or None if Wikipedia
    genuinely has no such page (redirects=1 already resolved). Raises FetchError if
    every retry failed -- rate-limited (429) retries respect Retry-After and don't
    count against max_retries as hard, since a sustained 429 storm is exactly the
    case worth waiting out rather than giving up on."""
    params = {
        "action": "query", "titles": title, "format": "json",
        "prop": "extracts", "explaintext": 1, "redirects": 1,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    data = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 5))
                time.sleep(max(retry_after, 2 ** min(attempt, 6)))
                continue
            if attempt == max_retries:
                raise FetchError(f"HTTP {e.code} fetching {title!r}")
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == max_retries:
                raise FetchError(f"network error fetching {title!r}")
            time.sleep(2 ** attempt)
    if data is None:
        raise FetchError(f"exhausted retries fetching {title!r}")
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid == "-1" or "missing" in page:
            return None
        return page.get("extract", "")
    return None


def fetch_and_clean(title, tokenizer):
    """Returns (status, passage_or_None, raw_extract_or_None). status is 'ok',
    'missing', 'short', or 'error'. raw_extract is returned alongside the cleaned
    passage whenever a page was found at all (even if too short to keep as a
    passage), so --raw_out captures everything fetched, not just what passed the
    length filter -- cheap to store, and means a future length-threshold change
    doesn't need a refetch either."""
    try:
        extract = fetch_extract(title)
    except FetchError:
        return "error", None, None
    if extract is None:
        return "missing", None, None
    cleaned = RAG_CHARSET_FILTER.sub("", extract)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < MIN_PASSAGE_CHARS:
        return "short", None, extract
    return "ok", truncate_to_token_budget(tokenizer, cleaned[:MAX_PASSAGE_CHARS]), extract


def run_pass(titles, tokenizer, concurrency, out_f, raw_f, label):
    """One pass over titles, writing successes to out_f/raw_f as they complete.
    Returns the list of titles that hit a FetchError (rate limit/network) this pass,
    for a retry pass -- these must never be silently counted as "missing"."""
    t0 = time.time()
    done = 0
    kept = 0
    counts = {"missing": 0, "short": 0, "error": 0}
    errored = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(fetch_and_clean, t, tokenizer): t for t in titles}
        for future in futures:
            title = futures[future]
            status, passage, raw_extract = future.result()
            done += 1
            if raw_extract is not None:
                raw_f.write(json.dumps({"title": title, "raw_extract": raw_extract}) + "\n")
                raw_f.flush()
            if status == "ok":
                out_f.write(json.dumps({"title": title, "passage": passage.replace("\n", " ")}) + "\n")
                out_f.flush()
                kept += 1
            else:
                counts[status] += 1
                if status == "error":
                    errored.append(title)

            if done % 200 == 0 or done == len(titles):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta_min = (len(titles) - done) / rate / 60
                print(f"  [{label}] {done}/{len(titles)} | kept {kept}, missing {counts['missing']}, "
                      f"too-short {counts['short']}, fetch-error {counts['error']} | "
                      f"{rate:.1f}/s, eta {eta_min:.1f} min", flush=True)

    print(f"[{label}] done in {time.time()-t0:.0f}s: kept {kept}, missing {counts['missing']}, "
          f"too-short {counts['short']}, fetch-error {counts['error']}", flush=True)
    return errored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--raw_out", type=str, required=True,
                         help="raw (un-cleaned, un-truncated) extracts, so a future "
                              "charset/length fix can reprocess without a live refetch")
    parser.add_argument("--concurrency", type=int, default=4,
                         help="kept conservative -- Wikipedia's API 429-rate-limits "
                              "this fetch at higher concurrency")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    args = parser.parse_args()

    with open(args.titles) as f:
        titles = [line.rstrip("\n") for line in f if line.strip()]
    print(f"fetching {len(titles)} titles ...", flush=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    with open(args.out, "w", encoding="utf-8") as out_f, \
            open(args.raw_out, "w", encoding="utf-8") as raw_f:
        remaining = titles
        pass_num = 1
        while remaining:
            errored = run_pass(remaining, tokenizer, args.concurrency, out_f, raw_f, f"pass {pass_num}")
            if not errored:
                break
            print(f"retrying {len(errored)} fetch-errors after a cooldown ...", flush=True)
            time.sleep(30)
            remaining = errored
            pass_num += 1
            if pass_num > 5:
                print(f"giving up after 5 passes -- {len(remaining)} titles never fetched "
                      f"(genuine network/rate issues, not counted as missing): {remaining}", flush=True)
                break

    print("all passes done", flush=True)


if __name__ == "__main__":
    main()
