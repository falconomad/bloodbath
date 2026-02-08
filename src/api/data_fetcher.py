
import os
from datetime import date, timedelta

import finnhub
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

client = finnhub.Client(api_key=FINNHUB_KEY)

def get_price_data(ticker, period="6mo", interval="1d"):
    return yf.download(ticker, period=period, interval=interval)


def get_company_news(ticker, lookback_days=7, limit=10):
    try:
        to_date = date.today()
        from_date = to_date - timedelta(days=lookback_days)
        news = client.company_news(
            ticker,
            _from=from_date.isoformat(),
            to=to_date.isoformat(),
        )
        headlines = [n.get("headline", "").strip() for n in news]
        return [h for h in headlines if h][:limit]
    except Exception:
        return []
