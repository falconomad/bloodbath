AI Stock Advisor Project

Features:
- Finnhub news integration
- Hugging Face FinBERT sentiment analysis
- Technical indicators
- Event detection (with upcoming earnings calendar boost)
- Complex scoring engine
- Streamlit dashboard

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
7. `pip install -r requirements.txt`
8. `streamlit run app.py`

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

Data/Label Diagnostics:

- `./.venv/bin/python scripts/model_diagnostics.py --trace logs/recommendation_trace.jsonl --horizons 5,10,20`
