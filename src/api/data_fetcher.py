import os
import random
import time
from pathlib import Path
from datetime import date, timedelta

import finnhub
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

client = finnhub.Client(api_key=FINNHUB_KEY)
_alpha_status_logged = False
_yf_rate_limit_logged = False
_alpha_rate_limit_logged = False
_alpha_error_logged = False
_finnhub_status_logged = False
_finnhub_rate_limit_logged = False
_finnhub_error_logged = False


def _cache_dir():
    base = Path(os.getenv("PRICE_CACHE_DIR", ".cache/price_data"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_path(ticker, period, interval):
    safe = str(ticker).replace("/", "_").replace("\\", "_").replace(".", "_").replace("-", "_")
    return _cache_dir() / f"{safe}_{period}_{interval}.pkl"


def _save_price_cache(ticker, period, interval, frame):
    if frame is None or frame.empty:
        return
    path = _cache_path(ticker, period, interval)
    frame.to_pickle(path)


def _load_price_cache(ticker, period, interval, max_age_hours=72):
    path = _cache_path(ticker, period, interval)
    if not path.exists():
        return pd.DataFrame()
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > (max_age_hours * 3600):
        return pd.DataFrame()
    try:
        cached = pd.read_pickle(path)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            print(f"[data] {ticker}: using cached price data rows={len(cached)}")
            return cached
    except Exception as exc:
        print(f"[data] {ticker}: cache load failed ({exc})")
    return pd.DataFrame()


def _is_rate_limited(exc):
    name = exc.__class__.__name__
    message = str(exc).lower()
    return name == "YFRateLimitError" or "too many requests" in message or "rate limit" in message


def _backoff_sleep(attempt, base_seconds=1.0):
    # Exponential backoff with jitter keeps retries from stampeding Yahoo endpoints.
    delay = base_seconds * (2 ** max(0, attempt - 1)) + random.uniform(0, 0.25)
    time.sleep(delay)


def _alpha_symbol(ticker):
    # Alpha Vantage uses dot notation for share classes (e.g. BRK.B).
    return str(ticker).strip().replace("-", ".")


def _period_to_days(period):
    mapping = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 731,
        "5y": 1827,
    }
    return mapping.get(period, 186)


def _trim_period(frame, period):
    if frame.empty:
        return frame
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=_period_to_days(period))
    out = frame[frame.index >= cutoff]
    return out if not out.empty else frame


def _finnhub_resolution(interval):
    if interval == "1d":
        return "D"
    return None


def _finnhub_window_unix(period):
    now = int(time.time())
    days = _period_to_days(period)
    start = now - (days * 24 * 60 * 60)
    return start, now


def _get_price_data_from_finnhub(ticker, period="6mo", interval="1d"):
    global _finnhub_status_logged, _finnhub_rate_limit_logged, _finnhub_error_logged

    if not FINNHUB_KEY:
        if not _finnhub_status_logged:
            print("[data] finnhub price fallback disabled (missing FINNHUB_API_KEY)")
            _finnhub_status_logged = True
        return pd.DataFrame()

    resolution = _finnhub_resolution(interval)
    if resolution is None:
        if not _finnhub_error_logged:
            print(f"[data] finnhub fallback skipped (unsupported interval={interval})")
            _finnhub_error_logged = True
        return pd.DataFrame()

    if not _finnhub_status_logged:
        print("[data] finnhub price fallback enabled")
        _finnhub_status_logged = True

    start, end = _finnhub_window_unix(period)
    try:
        payload = client.stock_candles(ticker, resolution, start, end)
    except Exception as exc:
        message = str(exc).lower()
        if ("429" in message or "rate limit" in message) and not _finnhub_rate_limit_logged:
            print("[data] finnhub rate limited; skipping finnhub fallback for this cycle")
            _finnhub_rate_limit_logged = True
        elif not _finnhub_error_logged:
            print(f"[data] finnhub request failed (sample ticker={ticker}): {exc}")
            _finnhub_error_logged = True
        return pd.DataFrame()

    if not isinstance(payload, dict):
        return pd.DataFrame()

    status = payload.get("s")
    if status != "ok":
        # no_data is common when window/symbol has no candles.
        if status == "no_data":
            return pd.DataFrame()
        if not _finnhub_error_logged:
            print(f"[data] finnhub bad status (sample ticker={ticker}, status={status})")
            _finnhub_error_logged = True
        return pd.DataFrame()

    timestamps = payload.get("t", [])
    if not timestamps:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "Open": payload.get("o", []),
            "High": payload.get("h", []),
            "Low": payload.get("l", []),
            "Close": payload.get("c", []),
            "Volume": payload.get("v", []),
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(how="all").sort_index()
    frame = _trim_period(frame, period)
    if not frame.empty:
        print(f"[data] {ticker}: finnhub rows={len(frame)}")
    return frame


def _get_price_data_from_alpha_vantage(ticker, period="6mo", interval="1d", max_retries=1):
    global _alpha_status_logged, _alpha_rate_limit_logged, _alpha_error_logged
    if not ALPHAVANTAGE_KEY:
        if not _alpha_status_logged:
            print("[data] alpha fallback disabled (missing ALPHAVANTAGE_API_KEY / ALPHA_VANTAGE_API_KEY)")
            _alpha_status_logged = True
        return pd.DataFrame()
    if not _alpha_status_logged:
        print("[data] alpha fallback enabled")
        _alpha_status_logged = True
    if interval != "1d":
        print(f"[data] {ticker}: alpha fallback skipped (unsupported interval={interval})")
        return pd.DataFrame()

    symbol = _alpha_symbol(ticker)
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": ALPHAVANTAGE_KEY,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(ALPHAVANTAGE_URL, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            print(f"[data] alpha request failed for {ticker} ({exc})")
            return pd.DataFrame()

        if isinstance(payload, dict):
            if payload.get("Note"):
                if not _alpha_rate_limit_logged:
                    print("[data] alpha rate limited; skipping alpha fallback for this cycle")
                    _alpha_rate_limit_logged = True
                return pd.DataFrame()
            if payload.get("Information") and not _alpha_error_logged:
                print(f"[data] alpha info: {payload.get('Information')}")
                _alpha_error_logged = True
                return pd.DataFrame()
            if payload.get("Error Message") and not _alpha_error_logged:
                print(f"[data] alpha error: {payload.get('Error Message')}")
                _alpha_error_logged = True
                return pd.DataFrame()

        series = payload.get("Time Series (Daily)") if isinstance(payload, dict) else None
        if not isinstance(series, dict) or not series:
            if not _alpha_error_logged:
                keys = list(payload.keys()) if isinstance(payload, dict) else []
                print(f"[data] alpha response missing daily series (sample ticker={ticker}, keys={keys})")
                _alpha_error_logged = True
            return pd.DataFrame()

        frame = pd.DataFrame.from_dict(series, orient="index")
        rename_map = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        }
        frame = frame.rename(columns=rename_map)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if any(col not in frame.columns for col in required):
            print(f"[data] {ticker}: alpha response missing required OHLCV columns")
            return pd.DataFrame()

        frame = frame[required]
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[frame.index.notna()]
        frame = frame.sort_index()

        for col in required:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        frame = frame.dropna(how="all")
        frame = _trim_period(frame, period)
        print(f"[data] {ticker}: alpha fallback rows={len(frame)}")
        return frame

    return pd.DataFrame()


def get_price_data(ticker, period="6mo", interval="1d", max_retries=1):
    global _yf_rate_limit_logged
    finnhub_data = _get_price_data_from_finnhub(ticker, period=period, interval=interval)
    if not finnhub_data.empty:
        _save_price_cache(ticker, period, interval, finnhub_data)
        return finnhub_data

    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if data.empty:
                if not _yf_rate_limit_logged:
                    print("[data] yfinance returned empty/rate-limited data; falling back to alpha/cache")
                    _yf_rate_limit_logged = True
                break
            print(f"[data] {ticker}: price rows={len(data)}")
            _save_price_cache(ticker, period, interval, data)
            return data
        except Exception as exc:
            if _is_rate_limited(exc):
                if not _yf_rate_limit_logged:
                    print("[data] yfinance rate limited; falling back to alpha/cache")
                    _yf_rate_limit_logged = True
                break
            print(f"[data] yfinance price fetch failed for {ticker} ({exc})")
            break

    alpha = _get_price_data_from_alpha_vantage(ticker, period=period, interval=interval, max_retries=2)
    if not alpha.empty:
        _save_price_cache(ticker, period, interval, alpha)
        return alpha

    cached = _load_price_cache(ticker, period, interval)
    if not cached.empty:
        return cached
    return pd.DataFrame()


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


def _fallback_single_ticker_fetch(tickers, period, interval):
    output = {}
    for ticker in tickers:
        output[ticker] = get_price_data(ticker, period=period, interval=interval, max_retries=1)
    empty_count = sum(1 for frame in output.values() if frame.empty)
    print(f"[data] single-ticker fallback complete: total={len(output)} empty={empty_count}")
    return output


def _fallback_alpha_or_cache_only(tickers, period, interval):
    output = {}
    for ticker in tickers:
        finnhub_data = _get_price_data_from_finnhub(ticker, period=period, interval=interval)
        if not finnhub_data.empty:
            _save_price_cache(ticker, period, interval, finnhub_data)
            output[ticker] = finnhub_data
            continue

        alpha = _get_price_data_from_alpha_vantage(ticker, period=period, interval=interval, max_retries=1)
        if not alpha.empty:
            _save_price_cache(ticker, period, interval, alpha)
            output[ticker] = alpha
            continue

        cached = _load_price_cache(ticker, period, interval)
        output[ticker] = cached if not cached.empty else pd.DataFrame()

    empty_count = sum(1 for frame in output.values() if frame.empty)
    print(f"[data] alpha/cache fallback complete: total={len(output)} empty={empty_count}")
    return output


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
            threads=False,
        )
    except Exception as exc:
        if _is_rate_limited(exc):
            print(f"[data] bulk price fetch rate limited for {len(tickers)} tickers; falling back to alpha/cache")
            return _fallback_alpha_or_cache_only(tickers, period, interval)
        print(f"[data] bulk price fetch failed for {len(tickers)} tickers ({exc})")
        return _fallback_single_ticker_fetch(tickers, period, interval)

    if frame.empty:
        print(f"[data] bulk price fetch returned empty frame for {len(tickers)} tickers; falling back to alpha/cache")
        return _fallback_alpha_or_cache_only(tickers, period, interval)

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
