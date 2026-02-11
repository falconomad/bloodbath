#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.sp500_list import TOP20_SECTOR, get_sp500_universe


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_symbol(s: str) -> str:
    return str(s).strip().upper().replace("/", "_").replace("\\", "_").replace(".", "_").replace("-", "_")


def _load_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    if args.symbol_file:
        rows = Path(args.symbol_file).read_text(encoding="utf-8").splitlines()
        return [x.strip().upper() for x in rows if x.strip()]
    if args.universe == "sp500":
        symbols = [str(x).upper() for x in get_sp500_universe()]
    else:
        symbols = [str(x).upper() for x in TOP20_SECTOR]
    # Always include broad market benchmarks for regime features.
    for b in ["SPY", "QQQ"]:
        if b not in symbols:
            symbols.append(b)
    return symbols


def _alpaca_base_url() -> str:
    raw = str(os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets/v2")).strip()
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw.lstrip('/')}"
    return raw.rstrip("/")


def _read_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df[df["time"].notna()]
    return df


def _last_ts(df: pd.DataFrame) -> datetime | None:
    if df.empty or "time" not in df.columns:
        return None
    ts = df["time"].max()
    if pd.isna(ts):
        return None
    return ts.to_pydatetime().astimezone(timezone.utc)


def _normalize_bars(bars: list[dict[str, Any]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(bars)
    required = {"t", "o", "h", "l", "c", "v"}
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    frame = frame.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    frame = frame[frame["time"].notna()]
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[["time", "open", "high", "low", "close", "volume"]].dropna().sort_values("time")
    return frame


@dataclass
class FetchStats:
    requests: int = 0
    symbols_fetched: int = 0
    rows_downloaded: int = 0


def _fetch_symbol_daily_bars(
    session: requests.Session,
    symbol: str,
    start_iso: str,
    end_iso: str,
    headers: dict[str, str],
    max_requests: int,
    sleep_sec: float,
    retries: int,
    stats: FetchStats,
) -> pd.DataFrame:
    base = _alpaca_base_url()
    url = f"{base}/stocks/{symbol.replace('-', '.')}/bars"
    params = {
        "timeframe": "1Day",
        "start": start_iso,
        "end": end_iso,
        "adjustment": "raw",
        "feed": "iex",
        "limit": 10000,
    }

    all_rows: list[dict[str, Any]] = []
    next_token = None
    while True:
        if stats.requests >= max_requests:
            print(f"[alpaca] request cap reached ({max_requests}); stopping further calls")
            break

        local_params = dict(params)
        if next_token:
            local_params["page_token"] = next_token

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = session.get(url, headers=headers, params=local_params, timeout=20)
                stats.requests += 1
                if resp.status_code == 429:
                    if attempt > retries:
                        print(f"[alpaca] {symbol}: rate limited (429), giving up after retries")
                        return _normalize_bars(all_rows)
                    backoff = sleep_sec * (2**attempt)
                    print(f"[alpaca] {symbol}: rate limited, retrying in {backoff:.2f}s")
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                payload = resp.json() if isinstance(resp.json(), dict) else {}
                rows = payload.get("bars", [])
                if isinstance(rows, list) and rows:
                    all_rows.extend(rows)
                next_token = payload.get("next_page_token")
                time.sleep(max(sleep_sec, 0.0))
                break
            except Exception as exc:
                if attempt > retries:
                    print(f"[alpaca] {symbol}: request failed ({exc})")
                    return _normalize_bars(all_rows)
                backoff = sleep_sec * (2**attempt)
                print(f"[alpaca] {symbol}: transient error ({exc}), retrying in {backoff:.2f}s")
                time.sleep(backoff)

        if not next_token:
            break

    out = _normalize_bars(all_rows)
    stats.symbols_fetched += 1
    stats.rows_downloaded += len(out)
    return out


def _merge_and_save(existing: pd.DataFrame, downloaded: pd.DataFrame, path: Path) -> tuple[int, int]:
    if existing.empty:
        merged = downloaded.copy()
    elif downloaded.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, downloaded], ignore_index=True)

    if merged.empty:
        return 0, 0

    merged = merged.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    before = len(existing)
    after = len(merged)

    path.parent.mkdir(parents=True, exist_ok=True)
    merged = merged.copy()
    merged["time"] = pd.to_datetime(merged["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    merged.to_csv(path, index=False)
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental Alpaca daily bar ingestion with free-tier-friendly rate limits")
    parser.add_argument("--universe", choices=["top20", "sp500"], default="top20")
    parser.add_argument("--symbols", default="", help="Comma list symbols override")
    parser.add_argument("--symbol-file", default="", help="Path with one symbol per line")
    parser.add_argument("--output-dir", default="data/external/alpaca_daily")
    parser.add_argument("--years", type=int, default=10, help="History lookback for first-time symbols")
    parser.add_argument("--sleep-sec", type=float, default=0.8, help="Delay between requests")
    parser.add_argument("--max-requests", type=int, default=60, help="Hard cap of API requests per run")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap for this run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    api_secret = os.getenv("ALPACA_API_SECRET", "").strip()
    if (not api_key or not api_secret) and not args.dry_run:
        raise SystemExit("ALPACA_API_KEY/ALPACA_API_SECRET required")

    symbols = _load_symbols(args)
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    output_dir = Path(args.output_dir)
    now = _utc_now()
    end_iso = _to_iso_z(now)
    default_start = now - timedelta(days=max(int(args.years), 1) * 365)

    session = requests.Session()
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}

    stats = FetchStats()
    summary = {"total_symbols": len(symbols), "skipped_up_to_date": 0, "updated": 0, "requests": 0, "rows_downloaded": 0}

    for symbol in symbols:
        safe = _safe_symbol(symbol)
        path = output_dir / f"{safe}.csv"
        existing = _read_existing_csv(path)
        last = _last_ts(existing)

        # Skip if we already have recent daily data.
        if last is not None and last.date() >= (now - timedelta(days=1)).date():
            summary["skipped_up_to_date"] += 1
            continue

        start_dt = (last + timedelta(days=1)) if last is not None else default_start
        start_iso = _to_iso_z(start_dt)

        if args.dry_run:
            print(f"[dry-run] {symbol}: would fetch {start_iso} -> {end_iso} into {path}")
            continue

        downloaded = _fetch_symbol_daily_bars(
            session=session,
            symbol=symbol,
            start_iso=start_iso,
            end_iso=end_iso,
            headers=headers,
            max_requests=int(args.max_requests),
            sleep_sec=float(args.sleep_sec),
            retries=int(args.retries),
            stats=stats,
        )

        before, after = _merge_and_save(existing, downloaded, path)
        if after > before:
            summary["updated"] += 1
        print(f"[alpaca] {symbol}: existing={before} downloaded={len(downloaded)} total={after}")

        if stats.requests >= int(args.max_requests):
            break

    summary["requests"] = stats.requests
    summary["rows_downloaded"] = stats.rows_downloaded
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
