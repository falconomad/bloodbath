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
4. `pip install -r requirements.txt`
5. `streamlit run app.py`
