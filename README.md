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
2. Optional fallback: add `ALPHAVANTAGE_API_KEY` (or `ALPHA_VANTAGE_API_KEY`) in `.env` to fetch prices when Yahoo is rate-limited.
3. Optional resilience cache: set `PRICE_CACHE_DIR` to reuse recently fetched OHLCV data when providers are throttled (defaults to `.cache/price_data`).
4. Optional trading profile: set `TRADE_MODE` to `NORMAL` (default) or `AGGRESSIVE`.
   - `NORMAL` defaults: `SIGNAL_BUY_THRESHOLD=1.0`, `SIGNAL_SELL_THRESHOLD=-1.0`, `TOP20_MIN_BUY_SCORE=0.75`
   - `AGGRESSIVE` defaults: `SIGNAL_BUY_THRESHOLD=0.55`, `SIGNAL_SELL_THRESHOLD=-0.7`, `TOP20_MIN_BUY_SCORE=0.45`
   - You can override any threshold directly with env vars if needed.
5. `pip install -r requirements.txt`
6. `streamlit run app.py`
