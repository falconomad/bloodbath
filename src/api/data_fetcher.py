
import yfinance as yf
import finnhub
import os
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

client = finnhub.Client(api_key=FINNHUB_KEY)

def get_price_data(ticker):
    return yf.download(ticker, period="6mo", interval="1d")


def get_company_news(ticker):
    try:
        news = client.company_news(ticker, _from="2025-01-01", to="2026-01-01")
        return [n["headline"] for n in news[:10]]
    except:
        return []
