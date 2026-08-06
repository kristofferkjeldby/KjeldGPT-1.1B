#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== waiting for v10 generation/judging to finish (PID 53673) ==="
while ps -p 53673 > /dev/null 2>&1; do sleep 10; done
echo "v10 whitebox run complete"

echo "=== blackbox rejudge v10 ==="
ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 rejudge_blackbox.py --source v10 --run_name v10_blackbox

echo "=== comparing v8 (threshold 2.0) vs v10 (threshold 3.3) on success+premise_corrected ==="
python3 - <<'PYEOF'
import json

def score(name):
    with open(f"runs/{name}_summary.json") as f:
        s = json.load(f)
    t = s["tally"]
    return t.get("success", 0) + t.get("premise_corrected", 0)

v8s = score("v8_blackbox")
v10s = score("v10_blackbox")
print(f"v8_blackbox (threshold 2.0) score={v8s}")
print(f"v10_blackbox (threshold 3.3) score={v10s}")
winner = "v10" if v10s > v8s else "v8"
threshold = 3.3 if winner == "v10" else 2.0
with open("/tmp/consolidation_result.json", "w") as f:
    json.dump({"winner": winner, "threshold": threshold, "v8_score": v8s, "v10_score": v10s}, f)
print(f"winner={winner} threshold={threshold}")
PYEOF

WINNER=$(python3 -c "import json;print(json.load(open('/tmp/consolidation_result.json'))['winner'])")
THRESHOLD=$(python3 -c "import json;print(json.load(open('/tmp/consolidation_result.json'))['threshold'])")
echo "WINNER=$WINNER THRESHOLD=$THRESHOLD"

if [ "$WINNER" = "v10" ]; then
  echo "=== promoting v10 -> v8 ==="
  rm -f runs/v8.jsonl runs/v8_raw.jsonl runs/v8_summary.json runs/v8_blackbox.jsonl runs/v8_blackbox_summary.json
  mv runs/v10.jsonl runs/v8.jsonl
  mv runs/v10_raw.jsonl runs/v8_raw.jsonl
  mv runs/v10_summary.json runs/v8_summary.json
  mv runs/v10_blackbox.jsonl runs/v8_blackbox.jsonl
  mv runs/v10_blackbox_summary.json runs/v8_blackbox_summary.json
  python3 -c "
import json
for f in ['runs/v8_summary.json', 'runs/v8_blackbox_summary.json']:
    d = json.load(open(f))
    d['run_name'] = 'v8'
    json.dump(d, open(f, 'w'), indent=2)
"
else
  echo "=== v8 (threshold 2.0) wins, discarding v10 ==="
  rm -f runs/v10.jsonl runs/v10_raw.jsonl runs/v10_summary.json runs/v10_blackbox.jsonl runs/v10_blackbox_summary.json
fi

echo "=== updating KjeldChat flagship files from settled v8 ==="
cp runs/v8_blackbox.jsonl runs/KjeldChat.jsonl
python3 -c "
import json
s = json.load(open('runs/v8_blackbox_summary.json'))
s['run_name'] = 'KjeldChat'
json.dump(s, open('runs/KjeldChat_summary.json', 'w'), indent=2)
"

echo "=== running llama blackbox test on settled v8 RAG config (min_context_score=$THRESHOLD) ==="
ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 qa_loop_llama.py --min_context_score "$THRESHOLD" --run_name llama_v1
ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 rejudge_blackbox.py --source llama_v1 --run_name llama_v1_blackbox

echo "=== regenerating plots ==="
python3 plot_model_comparison.py

echo "=== deleting raw files ==="
rm -f runs/v8_raw.jsonl runs/llama_v1_raw.jsonl

echo "PIPELINE COMPLETE winner=$WINNER threshold=$THRESHOLD"
