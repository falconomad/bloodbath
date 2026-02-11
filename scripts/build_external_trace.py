#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _norm(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    v = x / scale
    return max(-1.0, min(1.0, float(v)))


def _load_symbol_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    need = {"time", "open", "high", "low", "close", "volume"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
    return df


def _rows_from_bars(symbol: str, df: pd.DataFrame, min_rows: int = 40) -> list[dict]:
    if df.empty or len(df) < min_rows:
        return []

    close = df["close"]
    ret_1 = close.pct_change().fillna(0.0)
    ret_5 = close.pct_change(5).fillna(0.0)
    ma_5 = close.rolling(5).mean()
    ma_20 = close.rolling(20).mean()
    ma_ratio = ((ma_5 - ma_20) / ma_20).replace([pd.NA, pd.NaT], 0.0).fillna(0.0)
    vol_20 = ret_1.rolling(20).std().fillna(0.0)
    vol_rel = (df["volume"] / df["volume"].rolling(20).mean()).replace([pd.NA, pd.NaT], 1.0).fillna(1.0)

    out = []
    for i in range(len(df)):
        if i < 25:
            continue
        ts = df.loc[i, "time"]
        price = float(df.loc[i, "close"])
        if price <= 0:
            continue

        trend_v = _norm(float(ma_ratio.iloc[i]), 0.03)
        micro_v = _norm(float(ret_1.iloc[i]), 0.03)
        events_v = _norm(float(ret_5.iloc[i]), 0.08)
        volatility_v = -abs(_norm(float(vol_20.iloc[i]), 0.04))

        signals = {
            "trend": {"value": trend_v, "confidence": 0.8, "quality_ok": True},
            "sentiment": {"value": 0.0, "confidence": 0.0, "quality_ok": False},
            "social": {"value": 0.0, "confidence": 0.0, "quality_ok": False},
            "market": {"value": 0.0, "confidence": 0.3, "quality_ok": True},
            "events": {"value": events_v, "confidence": 0.5, "quality_ok": True},
            "micro": {"value": micro_v, "confidence": 0.6, "quality_ok": True},
            "dip": {"value": _norm(-float(ret_5.iloc[i]), 0.08), "confidence": 0.4, "quality_ok": True},
            "volatility": {"value": volatility_v, "confidence": 0.7, "quality_ok": True},
        }

        score = 0.45 * trend_v + 0.25 * micro_v + 0.20 * events_v + 0.10 * volatility_v
        out.append(
            {
                "ticker": symbol,
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "price": round(price, 6),
                "score": round(float(score), 6),
                "confidence": 0.55,
                "conflict_ratio": 0.2,
                "decision": "HOLD",
                "decision_reasons": ["external_history_feature_row"],
                "signals": signals,
                "risk_context": {
                    "rel_volume": float(max(0.0, vol_rel.iloc[i])),
                    "atr_pct": float(max(vol_20.iloc[i], 0.0)),
                    "data_gap_ratio": 0.0,
                    "portfolio_drawdown": 0.0,
                    "portfolio_avg_correlation": 0.0,
                    "ticker_sector_allocation": 0.0,
                    "sentiment_variance": 0.0,
                    "sentiment_article_count": 0,
                    "social_post_count": 0,
                    "market_news_count": 0,
                    "event_article_count": 0,
                    "market_regime": "external",
                },
                "source": "external_history",
            }
        )
    return out


def _market_features_by_time(input_dir: Path) -> dict[str, tuple[float, str]]:
    # Build a simple market context map keyed by day from SPY/QQQ if available.
    m: dict[str, tuple[float, str]] = {}
    for bench in ["SPY", "QQQ"]:
        p = input_dir / f"{bench}.csv"
        if not p.exists():
            p = input_dir / f"{bench.lower()}.csv"
        if not p.exists():
            continue
        df = _load_symbol_csv(p)
        if df.empty or len(df) < 25:
            continue
        close = df["close"]
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ret5 = close.pct_change(5).fillna(0.0)
        for i in range(25, len(df)):
            ts = df.loc[i, "time"]
            day = ts.strftime("%Y-%m-%d")
            trend = _norm(float(((ma5.iloc[i] - ma20.iloc[i]) / ma20.iloc[i]) if ma20.iloc[i] else 0.0), 0.03)
            if abs(trend) < 0.15:
                regime = "sideways"
            elif trend > 0:
                regime = "bull"
            else:
                regime = "bear"
            score = 0.7 * trend + 0.3 * _norm(float(ret5.iloc[i]), 0.08)
            prev = m.get(day)
            if prev is None:
                m[day] = (score, regime)
            else:
                m[day] = ((prev[0] + score) / 2.0, prev[1])
    return m


def _inject_market_context(rows: list[dict], market_map: dict[str, tuple[float, str]]) -> None:
    for row in rows:
        ts = str(row.get("ts", ""))
        day = ts[:10] if len(ts) >= 10 else ""
        market_score, regime = market_map.get(day, (0.0, "external"))
        signals = row.get("signals", {}) or {}
        market_sig = signals.get("market", {}) or {}
        market_sig["value"] = float(max(-1.0, min(1.0, market_score)))
        market_sig["confidence"] = 0.6 if day in market_map else 0.2
        market_sig["quality_ok"] = bool(day in market_map)
        signals["market"] = market_sig
        row["signals"] = signals
        risk = row.get("risk_context", {}) or {}
        risk["market_regime"] = str(regime)
        row["risk_context"] = risk


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trace-like training rows from external Alpaca daily CSVs")
    parser.add_argument("--input-dir", default="data/external/alpaca_daily")
    parser.add_argument("--output", default="artifacts/models/external_trace.jsonl")
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.csv")) if input_dir.exists() else []
    if args.max_symbols and args.max_symbols > 0:
        files = files[: args.max_symbols]

    rows = []
    market_map = _market_features_by_time(input_dir)
    for fp in files:
        symbol = fp.stem.replace("_", "-").upper()
        if symbol in {"SPY", "QQQ"}:
            continue
        df = _load_symbol_csv(fp)
        rows.extend(_rows_from_bars(symbol, df))
    _inject_market_context(rows, market_map)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(json.dumps({"status": "ok", "symbols": len(files), "rows": len(rows), "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
