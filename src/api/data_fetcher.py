import os
from datetime import date, timedelta

import finnhub
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

client = finnhub.Client(api_key=FINNHUB_KEY)


def get_price_data(ticker, period="6mo", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    print(f"[data] {ticker}: price rows={len(data)} empty={data.empty}")
    return data


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
        out = [h for h in headlines if h][:limit]
        print(f"[data] {ticker}: news_count={len(out)}")
        return out
    except Exception as exc:
        print(f"[data] {ticker}: company_news failed ({exc})")
        return []


def get_bulk_price_data(tickers, period="6mo", interval="1d"):
    """Fetch OHLCV data for multiple tickers in one yfinance call."""
    if not tickers:
        return {}

    try:
        frame = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print(f"[data] bulk price fetch failed for {len(tickers)} tickers ({exc})")
        return {ticker: pd.DataFrame() for ticker in tickers}

    if frame.empty:
        print(f"[data] bulk price fetch returned empty frame for {len(tickers)} tickers")
        return {ticker: pd.DataFrame() for ticker in tickers}

    # MultiIndex columns when multiple symbols are fetched.
    if isinstance(frame.columns, pd.MultiIndex):
        output = {}
        for ticker in tickers:
            if ticker in frame.columns.get_level_values(0):
                data = frame[ticker].dropna(how="all")
                output[ticker] = data
            else:
                output[ticker] = pd.DataFrame()
        empty_count = sum(1 for d in output.values() if d.empty)
        print(f"[data] bulk price fetch complete: total={len(output)} empty={empty_count}")
        return output

    # Single ticker fallback shape.
    ticker = tickers[0]
    output = {ticker: frame.dropna(how="all")}
    print(f"[data] bulk single ticker fetch: {ticker} rows={len(output[ticker])} empty={output[ticker].empty}")
    return output


def get_earnings_calendar(ticker, lookahead_days=30):
    start_date = date.today()
    end_date = start_date + timedelta(days=lookahead_days)

    try:
        payload = client.earnings_calendar(
            _from=start_date.isoformat(),
            to=end_date.isoformat(),
            symbol=ticker,
        )
        calendar = payload.get("earningsCalendar", []) if isinstance(payload, dict) else []
        out = calendar if isinstance(calendar, list) else []
        print(f"[data] {ticker}: earnings_count={len(out)}")
        return out
    except Exception:
        if not FINNHUB_KEY:
            print(f"[data] {ticker}: earnings_calendar skipped (missing FINNHUB_API_KEY)")
            return []

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
                "symbol": ticker,
                "token": FINNHUB_KEY,
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        calendar = payload.get("earningsCalendar", []) if isinstance(payload, dict) else []
        out = calendar if isinstance(calendar, list) else []
        print(f"[data] {ticker}: earnings_count={len(out)}")
        return out
    except Exception as exc:
        print(f"[data] {ticker}: earnings_calendar failed ({exc})")
        return []
