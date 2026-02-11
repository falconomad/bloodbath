#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import finnhub
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.sp500_list import TOP20_SECTOR, get_sp500_universe


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    for b in ["SPY", "QQQ"]:
        if b not in symbols:
            symbols.append(b)
    return symbols


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


def _last_ts(df: pd.DataFrame):
    if df.empty or "time" not in df.columns:
        return None
    ts = df["time"].max()
    if pd.isna(ts):
        return None
    return ts.to_pydatetime().astimezone(timezone.utc)


def _fetch_finnhub_daily(client: finnhub.Client, symbol: str, start_unix: int, end_unix: int) -> pd.DataFrame:
    payload = client.stock_candles(symbol, "D", start_unix, end_unix)
    if not isinstance(payload, dict) or payload.get("s") != "ok":
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    t = payload.get("t", [])
    if not t:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(t, unit="s", utc=True),
            "open": payload.get("o", []),
            "high": payload.get("h", []),
            "low": payload.get("l", []),
            "close": payload.get("c", []),
            "volume": payload.get("v", []),
        }
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "close"]).sort_values("time")
    return df


def _merge_and_save(existing: pd.DataFrame, downloaded: pd.DataFrame, path: Path):
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
    parser = argparse.ArgumentParser(description="Incremental Finnhub daily bar ingestion with free-tier-friendly limits")
    parser.add_argument("--universe", choices=["top20", "sp500"], default="top20")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbol-file", default="")
    parser.add_argument("--output-dir", default="data/external/finnhub_daily")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--sleep-sec", type=float, default=1.1)
    parser.add_argument("--max-requests", type=int, default=45)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit("FINNHUB_API_KEY required")

    symbols = _load_symbols(args)
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    output_dir = Path(args.output_dir)
    now = _utc_now()
    default_start = now - timedelta(days=max(int(args.years), 1) * 365)

    summary = {"total_symbols": len(symbols), "skipped_up_to_date": 0, "updated": 0, "requests": 0}

    client = finnhub.Client(api_key=api_key) if api_key else None

    for symbol in symbols:
        if summary["requests"] >= int(args.max_requests):
            print(f"[finnhub] request cap reached ({args.max_requests}); stopping")
            break

        path = output_dir / f"{_safe_symbol(symbol)}.csv"
        existing = _read_existing_csv(path)
        last = _last_ts(existing)

        if last is not None and last.date() >= (now - timedelta(days=1)).date():
            summary["skipped_up_to_date"] += 1
            continue

        start_dt = (last + timedelta(days=1)) if last is not None else default_start
        start_unix = int(start_dt.timestamp())
        end_unix = int(now.timestamp())

        if args.dry_run:
            print(f"[dry-run] {symbol}: would fetch {start_dt.isoformat()} -> {now.isoformat()} into {path}")
            continue

        try:
            downloaded = _fetch_finnhub_daily(client, symbol, start_unix, end_unix)
            summary["requests"] += 1
        except Exception as exc:
            print(f"[finnhub] {symbol}: fetch failed ({exc})")
            time.sleep(float(args.sleep_sec) * 2)
            continue

        before, after = _merge_and_save(existing, downloaded, path)
        if after > before:
            summary["updated"] += 1
        print(f"[finnhub] {symbol}: existing={before} downloaded={len(downloaded)} total={after}")
        time.sleep(max(float(args.sleep_sec), 0.0))

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
