Bloodbath Engine (Headless)

This repository now runs a headless trading engine:
- pulls market/news context from Alpaca
- computes local technical + sentiment pre-scores
- sends only shortlisted symbols to Gemini for final decision
- validates through risk manager
- executes orders via Alpaca API

Run:

1. Set env vars:
   - `ALPACA_API_KEY`
   - `ALPACA_API_SECRET`
   - `GEMINI_API_KEY`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Execute:
   - `python main.py`

Gemini optimization knobs:

- `MAX_GEMINI_CALLS_PER_RUN` (default `3`)
- `PRE_FILTER_MIN_TECH_SCORE` (default `55`)
- `PRE_FILTER_MIN_SENTIMENT_SCORE` (default `45`)
- `PRE_FILTER_MIN_LOCAL_SCORE` (default `60`)
- `PRE_FILTER_MIN_CONFIDENCE` (default `0.55`)
- `API_SLEEP_SECONDS` (default `15`)
- `GEMINI_MIN_SECONDS_BETWEEN_CALLS` (default `15`)
- `GEMINI_MAX_RETRIES_ON_429` (default `1`)
- `GEMINI_MODEL` (default `gemini-2.5-flash`)
- `GEMINI_QUOTA_STATE_PATH` (default `logs/gemini_quota_state.json`)
- `ENGINE_EVENTS_PATH` (default `logs/engine_events.jsonl`)

Notes:
- Engine uses deterministic fallbacks if Gemini returns `429 RESOURCE_EXHAUSTED`.
- Risk manager still enforces allocation caps and macro crash protection before execution.
- Candidate selection is now regime-aware and confidence-aware before Gemini.
- Full run telemetry is emitted as JSONL for tuning and post-trade analysis.

Fresh Start (Reset State + Set Capital):

- `./.venv/bin/python scripts/fresh_start.py 5000`
- This resets persisted trading state tables, seeds the portfolio to the provided amount, and updates `.env` with `STARTING_CAPITAL`.

Predictive Model Training (Next 5/10 Step Returns):

- `./.venv/bin/python -m src.ml.predictive_model --trace logs/recommendation_trace.jsonl --horizon 5 --model random_forest`
- `./.venv/bin/python -m src.ml.predictive_model --trace logs/recommendation_trace.jsonl --horizon 10 --model gradient_boosting`
- Save model artifact for live inference:
  - `./.venv/bin/python -m src.ml.predictive_model --trace logs/recommendation_trace.jsonl --horizon 5 --model random_forest --save-artifact artifacts/models/return_model.pkl`
- Outputs classification metrics (`precision`, `recall`, `roc_auc`) and profit metrics (`strategy_total_return`, `profit_factor`, `avg_trade_expectancy`).
- Live engine auto-uses ML score when a model artifact is found at `artifacts/models/return_model.pkl` (or `artifacts/return_model.pkl`), and falls back to heuristic scoring if no artifact is available.

Model Retraining Loop:

- `./scripts/retrain_model.sh`
- If `DATABASE_URL` is set, it refreshes trace data from DB, trains a candidate model, and promotes it only if metrics improve.
- Force promote a retrain run:
  - `./scripts/retrain_model.sh --force`
- The retrain script now searches multiple candidates automatically:
  - horizons: `5,10,20`
  - models: `random_forest,gradient_boosting`
- It now skips promotion automatically when holdout positive-rate is zero (to avoid promoting models trained on a no-signal window).
- It now auto-merges external Alpaca history (if present in `data/external/alpaca_daily`) with DB trace before training.
- Disable external merge if needed:
  - `./scripts/retrain_model.sh --no-external`
- Promotion gates (default):
  - `positive_rate_test >= 0.08`
  - `roc_auc >= 0.53`
  - `profit_factor >= 1.05`
  - `trades >= 20`
- Override gates if needed:
  - `./scripts/retrain_model.sh --min-pos-rate 0.05 --min-roc-auc 0.52 --min-profit-factor 1.02 --min-trades 10`

Data/Label Diagnostics:

- `./.venv/bin/python scripts/model_diagnostics.py --trace logs/recommendation_trace.jsonl --horizons 5,10,20`

Alpaca External Data Ingestion (Free-Tier Safe):

- `./.venv/bin/python scripts/ingest_alpaca_history.py --universe top20 --sleep-sec 0.8 --max-requests 60`
- Incremental behavior:
  - per symbol, skips API call if local file is already up-to-date
  - fetches only missing window from last saved date forward
- Test plan without credentials/calls:
  - `./.venv/bin/python scripts/ingest_alpaca_history.py --universe top20 --max-symbols 5 --dry-run`

Finnhub External Data Ingestion (Free-Tier Safe):

- `./.venv/bin/python scripts/ingest_finnhub_history.py --universe top20 --sleep-sec 1.1 --max-requests 45`
- Test plan without credentials/calls:
  - `./.venv/bin/python scripts/ingest_finnhub_history.py --universe top20 --max-symbols 5 --dry-run`
# Placeholder
