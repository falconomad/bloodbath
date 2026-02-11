#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

TRACE_PATH="$ROOT_DIR/logs/recommendation_trace.jsonl"
MODEL_DIR="$ROOT_DIR/artifacts/models"
MODEL_PATH="$MODEL_DIR/return_model.pkl"
METRICS_PATH="$MODEL_DIR/return_model.metrics.json"
CANDIDATE_PATH="$MODEL_DIR/return_model.candidate.pkl"
CANDIDATE_METRICS_PATH="$MODEL_DIR/return_model.candidate.metrics.json"

HORIZON=5
MODEL_FAMILY="random_forest"
SEARCH_HORIZONS="5,10,20"
SEARCH_MODELS="random_forest,gradient_boosting"
EXPORT_DB=1
FORCE_PROMOTE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --horizon N            Forward horizon (default: 5)
  --model NAME           Model family: random_forest|gradient_boosting (default: random_forest)
  --horizons CSV         Search horizons (default: 5,10,20)
  --models CSV           Search model families (default: random_forest,gradient_boosting)
  --trace PATH           Trace jsonl path (default: logs/recommendation_trace.jsonl)
  --no-export-db         Do not refresh trace from DB
  --force                Promote candidate even if not better
  -h, --help             Show this help

Notes:
  - If DATABASE_URL is set and export is enabled, trace is refreshed from recommendation_trace table.
  - Candidate model is promoted only when metrics improve unless --force is used.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --horizon)
      HORIZON="$2"; shift 2 ;;
    --model)
      MODEL_FAMILY="$2"; shift 2 ;;
    --horizons)
      SEARCH_HORIZONS="$2"; shift 2 ;;
    --models)
      SEARCH_MODELS="$2"; shift 2 ;;
    --trace)
      TRACE_PATH="$2"; shift 2 ;;
    --no-export-db)
      EXPORT_DB=0; shift ;;
    --force)
      FORCE_PROMOTE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

mkdir -p "$MODEL_DIR"
mkdir -p "$(dirname "$TRACE_PATH")"

if [[ "$EXPORT_DB" -eq 1 && -n "${DATABASE_URL:-}" ]]; then
  echo "[retrain] exporting trace rows from DB -> $TRACE_PATH"
  DATABASE_URL="$DATABASE_URL" TRACE_PATH="$TRACE_PATH" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
import psycopg2

url = os.environ["DATABASE_URL"]
out_path = Path(os.environ.get("TRACE_PATH"))
out_path.parent.mkdir(parents=True, exist_ok=True)

conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute("SELECT payload_json FROM recommendation_trace WHERE payload_json IS NOT NULL ORDER BY id ASC")
rows = cur.fetchall() or []
cur.close()
conn.close()

count = 0
with out_path.open("w", encoding="utf-8") as f:
    for (payload_json,) in rows:
        text = str(payload_json or "").strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")
            count += 1
        except Exception:
            continue

print(f"EXPORTED_ROWS={count}")
PY
elif [[ "$EXPORT_DB" -eq 1 ]]; then
  echo "[retrain] DATABASE_URL not set; using existing trace file"
fi

if [[ ! -s "$TRACE_PATH" ]]; then
  echo "[retrain] trace file missing or empty: $TRACE_PATH" >&2
  exit 1
fi

echo "[retrain] training candidate model"
"$PYTHON_BIN" -m src.ml.predictive_model \
  --trace "$TRACE_PATH" \
  --search \
  --horizons "$SEARCH_HORIZONS" \
  --models "$SEARCH_MODELS" \
  --horizon "$HORIZON" \
  --model "$MODEL_FAMILY" \
  --save-artifact "$CANDIDATE_PATH" \
  --output "$CANDIDATE_METRICS_PATH"

if [[ ! -f "$CANDIDATE_METRICS_PATH" ]]; then
  echo "[retrain] missing candidate metrics output" >&2
  exit 1
fi

STATUS="$($PYTHON_BIN - <<PY
import json
from pathlib import Path
obj = json.loads(Path("$CANDIDATE_METRICS_PATH").read_text(encoding="utf-8"))
saved = obj.get("best_saved") or {}
print(saved.get("status", obj.get("status", "")))
PY
)"

if [[ "$STATUS" != "ok" ]]; then
  echo "[retrain] training did not produce a usable model (status=$STATUS)"
  rm -f "$CANDIDATE_PATH"
  exit 1
fi

if [[ ! -f "$CANDIDATE_PATH" ]]; then
  echo "[retrain] candidate model file missing: $CANDIDATE_PATH" >&2
  exit 1
fi

PROMOTE=1
REASON="first_model"

if [[ -f "$MODEL_PATH" && -f "$METRICS_PATH" && "$FORCE_PROMOTE" -ne 1 ]]; then
  COMPARE="$($PYTHON_BIN - <<PY
import json
from pathlib import Path

new = json.loads(Path("$CANDIDATE_METRICS_PATH").read_text(encoding="utf-8"))
old = json.loads(Path("$METRICS_PATH").read_text(encoding="utf-8"))

def getvals(obj):
    cand = (obj.get("best_saved") or obj)
    p = float((((cand.get("profit") or {}).get("profit_factor", 0.0)) or 0.0))
    a = float((((cand.get("classification") or {}).get("roc_auc", 0.0)) or 0.0))
    e = float((((cand.get("profit") or {}).get("avg_trade_expectancy", 0.0)) or 0.0))
    return (p, a, e)

n = getvals(new)
o = getvals(old)
if n > o:
    print("PROMOTE")
    print(f"new={n} old={o}")
else:
    print("KEEP")
    print(f"new={n} old={o}")
PY
)"
  DECISION="$(echo "$COMPARE" | head -n1)"
  DETAIL="$(echo "$COMPARE" | tail -n1)"
  if [[ "$DECISION" == "KEEP" ]]; then
    PROMOTE=0
    REASON="$DETAIL"
  else
    REASON="$DETAIL"
  fi
fi

if [[ "$FORCE_PROMOTE" -eq 1 ]]; then
  PROMOTE=1
  REASON="forced"
fi

if [[ "$PROMOTE" -eq 1 ]]; then
  if [[ -f "$MODEL_PATH" ]]; then
    cp "$MODEL_PATH" "$MODEL_DIR/return_model.prev.pkl"
  fi
  if [[ -f "$METRICS_PATH" ]]; then
    cp "$METRICS_PATH" "$MODEL_DIR/return_model.prev.metrics.json"
  fi
  mv "$CANDIDATE_PATH" "$MODEL_PATH"
  mv "$CANDIDATE_METRICS_PATH" "$METRICS_PATH"
  echo "[retrain] promoted new model -> $MODEL_PATH ($REASON)"
else
  rm -f "$CANDIDATE_PATH"
  rm -f "$CANDIDATE_METRICS_PATH"
  echo "[retrain] kept existing model ($REASON)"
fi

echo "[retrain] done"
