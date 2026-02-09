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
5. Optional rotating fetch: `FETCH_BATCH_SIZE` (default `20`, which means no rotation for TOP20).
   - Example: set `FETCH_BATCH_SIZE=5` to fetch 5 symbols per run and rotate chunks.
6. `pip install -r requirements.txt`
7. `streamlit run app.py`
