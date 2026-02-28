AI Stock Advisor Project

Features:
- Finnhub news integration
- Hugging Face FinBERT sentiment analysis
- Technical indicators
- Event detection (with upcoming earnings calendar boost)
- Complex scoring engine
- Streamlit dashboard
- Goal-driven agent pacing (capital target + deadline)
- React dashboard (Netlify-ready) + Python API

Setup:

1. Add `FINNHUB_API_KEY` in `.env`.
2. Optional fallback: add `ALPACA_API_KEY` and `ALPACA_API_SECRET` to use Alpaca market data as a secondary price source.
   - Optional: `ALPACA_DATA_BASE_URL` (default `https://data.alpaca.markets/v2`)
3. Optional resilience cache: set `PRICE_CACHE_DIR` to reuse recently fetched OHLCV data when providers are throttled (defaults to `.cache/price_data`).
4. Optional trading profile: set `TRADE_MODE` to `NORMAL` (default) or `AGGRESSIVE`.
   - `NORMAL` defaults: `SIGNAL_BUY_THRESHOLD=1.0`, `SIGNAL_SELL_THRESHOLD=-1.0`, `TOP20_MIN_BUY_SCORE=0.75`
   - `AGGRESSIVE` defaults: `SIGNAL_BUY_THRESHOLD=0.55`, `SIGNAL_SELL_THRESHOLD=-0.7`, `TOP20_MIN_BUY_SCORE=0.45`
   - You can override any threshold directly with env vars if needed.
5. Optional execution realism:
   - `TOP20_SLIPPAGE_BPS` (default `5.0`)
   - `TOP20_FEE_BPS` (default `1.0`)
   These are applied to buys/sells in simulation.
6. Optional rotating fetch: `FETCH_BATCH_SIZE` (default `20`, which means no rotation for TOP20).
   - Example: set `FETCH_BATCH_SIZE=5` to fetch 5 symbols per run and rotate chunks.
7. Configure objective in `config/agent_goal.yaml` (or env overrides):
   - `start_capital`
   - `target_capital`
   - `horizon_days`
   - `start_date` (optional ISO-8601)
8. Goal env overrides (optional):
   - `AGENT_GOAL_START_CAPITAL`
   - `AGENT_GOAL_TARGET_CAPITAL`
   - `AGENT_GOAL_HORIZON_DAYS`
   - `AGENT_GOAL_START_DATE`
9. API backend:
   - `pip install -r requirements.txt`
   - `uvicorn api.server:app --host 0.0.0.0 --port 8000`
10. React frontend:
   - `cd frontend/react`
   - `npm install`
   - `cp .env.example .env` and set `VITE_API_BASE_URL`
   - `npm run dev`
11. Legacy Streamlit dashboard:
   - `streamlit run frontend/streamlit_app.py`
   - `streamlit run app.py` (legacy entrypoint)

Netlify deploy (React app):

- Base directory: `frontend/react`
- Build command: `npm run build`
- Publish directory: `dist`
- Environment variable:
  - `VITE_API_BASE_URL=https://<your-backend-domain>`

Backend endpoints (summary):

- `GET /health`
- `GET /api/portfolio`
- `GET /api/transactions`
- `GET /api/positions`
- `GET /api/signals`
- `GET /api/goal/latest`
- `GET /api/manual-checks`
- `POST /api/manual-checks`
- `DELETE /api/manual-checks/{ticker}`
- `POST /api/cycle/run`

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
