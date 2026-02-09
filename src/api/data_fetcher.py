import os
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
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")


def _env_non_empty(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


ALPACA_DATA_BASE_URL = _env_non_empty("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets/v2")

client = finnhub.Client(api_key=FINNHUB_KEY)
_yf_rate_limit_logged = False
_finnhub_status_logged = False
_finnhub_rate_limit_logged = False
_finnhub_error_logged = False
_alpaca_status_logged = False
_alpaca_rate_limit_logged = False
_alpaca_error_logged = False
_alpaca_snapshot_logged = False


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


def _alpaca_symbol(ticker):
    # Alpaca market data uses dot notation for share classes.
    return str(ticker).strip().replace("-", ".")


def _normalized_alpaca_base_url():
    base = str(ALPACA_DATA_BASE_URL).strip()
    if not base:
        base = "https://data.alpaca.markets/v2"
    if not base.startswith(("http://", "https://")):
        base = f"https://{base.lstrip('/')}"
    return base.rstrip("/")


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
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=_period_to_days(period))
    if getattr(frame.index, "tz", None) is None:
        cutoff = cutoff.tz_localize(None)
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


def _get_price_data_from_alpaca(ticker, period="6mo", interval="1d"):
    global _alpaca_status_logged, _alpaca_rate_limit_logged, _alpaca_error_logged

    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        if not _alpaca_status_logged:
            print("[data] alpaca price fallback disabled (missing ALPACA_API_KEY / ALPACA_API_SECRET)")
            _alpaca_status_logged = True
        return pd.DataFrame()

    if interval != "1d":
        if not _alpaca_error_logged:
            print(f"[data] alpaca fallback skipped (unsupported interval={interval})")
            _alpaca_error_logged = True
        return pd.DataFrame()

    if not _alpaca_status_logged:
        print("[data] alpaca price fallback enabled")
        _alpaca_status_logged = True

    start_ts, end_ts = _finnhub_window_unix(period)
    start_iso = pd.to_datetime(start_ts, unit="s", utc=True).isoformat().replace("+00:00", "Z")
    end_iso = pd.to_datetime(end_ts, unit="s", utc=True).isoformat().replace("+00:00", "Z")

    symbol = _alpaca_symbol(ticker)
    url = f"{_normalized_alpaca_base_url()}/stocks/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    params = {
        "timeframe": "1Day",
        "start": start_iso,
        "end": end_iso,
        "adjustment": "raw",
        "limit": 10000,
        "feed": "iex",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code == 429:
            if not _alpaca_rate_limit_logged:
                print("[data] alpaca rate limited; skipping alpaca fallback for this cycle")
                _alpaca_rate_limit_logged = True
            return pd.DataFrame()
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        if not _alpaca_error_logged:
            print(f"[data] alpaca request failed (sample ticker={ticker}): {exc}")
            _alpaca_error_logged = True
        return pd.DataFrame()

    bars = payload.get("bars") if isinstance(payload, dict) else None
    if not isinstance(bars, list) or not bars:
        return pd.DataFrame()

    frame = pd.DataFrame(bars)
    required = ["o", "h", "l", "c", "v", "t"]
    if any(col not in frame.columns for col in required):
        if not _alpaca_error_logged:
            print(f"[data] alpaca response missing required fields (sample ticker={ticker})")
            _alpaca_error_logged = True
        return pd.DataFrame()

    frame = frame.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "time"})
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame[frame["time"].notna()].set_index("time")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all").sort_index()
    frame = _trim_period(frame, period)
    if not frame.empty:
        print(f"[data] {ticker}: alpaca rows={len(frame)}")
    return frame


def get_alpaca_snapshot_features(ticker):
    global _alpaca_snapshot_logged
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        return {"available": False, "spread_bps": 0.0, "intraday_return": 0.0, "rel_volume": 1.0, "quality": 0.0}

    symbol = _alpaca_symbol(ticker)
    url = f"{_normalized_alpaca_base_url()}/stocks/{symbol}/snapshot"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    params = {"feed": "iex"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json() if isinstance(response.json(), dict) else {}
    except Exception:
        return {"available": False, "spread_bps": 0.0, "intraday_return": 0.0, "rel_volume": 1.0, "quality": 0.0}

    if not _alpaca_snapshot_logged:
        print("[data] alpaca snapshot features enabled")
        _alpaca_snapshot_logged = True

    latest_quote = payload.get("latestQuote", {}) if isinstance(payload, dict) else {}
    daily_bar = payload.get("dailyBar", {}) if isinstance(payload, dict) else {}
    prev_daily_bar = payload.get("prevDailyBar", {}) if isinstance(payload, dict) else {}

    ask = float(latest_quote.get("ap", 0.0) or 0.0)
    bid = float(latest_quote.get("bp", 0.0) or 0.0)
    mid = (ask + bid) / 2.0 if (ask > 0 and bid > 0) else 0.0
    spread_bps = ((ask - bid) / mid) * 10_000.0 if mid > 0 and ask >= bid else 0.0

    open_p = float(daily_bar.get("o", 0.0) or 0.0)
    close_p = float(daily_bar.get("c", 0.0) or 0.0)
    intraday_return = ((close_p - open_p) / open_p) if open_p > 0 else 0.0

    today_v = float(daily_bar.get("v", 0.0) or 0.0)
    prev_v = float(prev_daily_bar.get("v", 0.0) or 0.0)
    rel_volume = (today_v / prev_v) if prev_v > 0 else 1.0

    # Higher quality when spreads are tighter and volumes are healthy.
    spread_quality = max(0.0, min(1.0, 1.0 - (spread_bps / 80.0)))
    vol_quality = max(0.0, min(1.0, rel_volume / 1.5))
    quality = (0.65 * spread_quality) + (0.35 * vol_quality)

    return {
        "available": True,
        "spread_bps": round(spread_bps, 4),
        "intraday_return": round(intraday_return, 5),
        "rel_volume": round(rel_volume, 4),
        "quality": round(float(quality), 4),
    }


def get_price_data(ticker, period="6mo", interval="1d", max_retries=1):
    global _yf_rate_limit_logged
    finnhub_data = _get_price_data_from_finnhub(ticker, period=period, interval=interval)
    if not finnhub_data.empty:
        _save_price_cache(ticker, period, interval, finnhub_data)
        return finnhub_data

    alpaca_data = _get_price_data_from_alpaca(ticker, period=period, interval=interval)
    if not alpaca_data.empty:
        _save_price_cache(ticker, period, interval, alpaca_data)
        return alpaca_data

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
                    print("[data] yfinance returned empty/rate-limited data; falling back to cache")
                    _yf_rate_limit_logged = True
                break
            print(f"[data] {ticker}: price rows={len(data)}")
            _save_price_cache(ticker, period, interval, data)
            return data
        except Exception as exc:
            if _is_rate_limited(exc):
                if not _yf_rate_limit_logged:
                    print("[data] yfinance rate limited; falling back to cache")
                    _yf_rate_limit_logged = True
                break
            print(f"[data] yfinance price fetch failed for {ticker} ({exc})")
            break

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


def _fallback_providers_or_cache_only(tickers, period, interval):
    output = {}
    for ticker in tickers:
        finnhub_data = _get_price_data_from_finnhub(ticker, period=period, interval=interval)
        if not finnhub_data.empty:
            _save_price_cache(ticker, period, interval, finnhub_data)
            output[ticker] = finnhub_data
            continue

        alpaca_data = _get_price_data_from_alpaca(ticker, period=period, interval=interval)
        if not alpaca_data.empty:
            _save_price_cache(ticker, period, interval, alpaca_data)
            output[ticker] = alpaca_data
            continue

        cached = _load_price_cache(ticker, period, interval)
        output[ticker] = cached if not cached.empty else pd.DataFrame()

    empty_count = sum(1 for frame in output.values() if frame.empty)
    print(f"[data] provider/cache fallback complete: total={len(output)} empty={empty_count}")
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
            print(f"[data] bulk price fetch rate limited for {len(tickers)} tickers; falling back to providers/cache")
            return _fallback_providers_or_cache_only(tickers, period, interval)
        print(f"[data] bulk price fetch failed for {len(tickers)} tickers ({exc})")
        return _fallback_single_ticker_fetch(tickers, period, interval)

    if frame.empty:
        print(f"[data] bulk price fetch returned empty frame for {len(tickers)} tickers; falling back to providers/cache")
        return _fallback_providers_or_cache_only(tickers, period, interval)

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
