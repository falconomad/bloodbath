from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any

from src.common.trace_utils import load_jsonl_dict_rows, safe_float


@dataclass
class TradeOutcome:
    ticker: str
    ts: str
    decision: str
    ret: float
    duration_steps: int


def load_trace_entries(trace_path: str) -> list[dict[str, Any]]:
    return load_jsonl_dict_rows(trace_path)


def build_trade_outcomes(entries: list[dict[str, Any]], horizon: int = 1) -> list[TradeOutcome]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        ticker = str(e.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        grouped.setdefault(ticker, []).append(e)

    outcomes: list[TradeOutcome] = []
    h = max(int(horizon), 1)
    for ticker, rows in grouped.items():
        rows.sort(key=lambda x: str(x.get("ts", "")))
        for i, row in enumerate(rows):
            j = i + h
            if j >= len(rows):
                continue
            decision = str(row.get("decision", "HOLD")).upper().strip()
            if decision not in {"BUY", "SELL"}:
                continue

            p0 = safe_float(row.get("price", None), default=float("nan"))
            p1 = safe_float(rows[j].get("price", None), default=float("nan"))
            if not (math.isfinite(p0) and math.isfinite(p1)) or p0 <= 0 or p1 <= 0:
                continue

            raw_ret = (p1 - p0) / p0
            realized = raw_ret if decision == "BUY" else -raw_ret
            outcomes.append(
                TradeOutcome(
                    ticker=ticker,
                    ts=str(row.get("ts", "")),
                    decision=decision,
                    ret=float(realized),
                    duration_steps=h,
                )
            )

    outcomes.sort(key=lambda x: x.ts)
    return outcomes


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _profit_factor(returns: list[float]) -> float:
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = -sum(r for r in returns if r < 0)
    if gross_loss <= 1e-12:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x)
    deny = sum((b - my) ** 2 for b in y)
    den = math.sqrt(max(denx * deny, 0.0))
    if den <= 1e-12:
        return 0.0
    return num / den


def module_contribution_report(entries: list[dict[str, Any]], horizon: int = 1) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        ticker = str(e.get("ticker", "")).upper().strip()
        if ticker:
            grouped.setdefault(ticker, []).append(e)

    signal_hist: dict[str, list[float]] = {}
    realized_hist: list[float] = []
    h = max(int(horizon), 1)

    for _, rows in grouped.items():
        rows.sort(key=lambda x: str(x.get("ts", "")))
        for i, row in enumerate(rows):
            j = i + h
            if j >= len(rows):
                continue
            p0 = safe_float(row.get("price", None), default=float("nan"))
            p1 = safe_float(rows[j].get("price", None), default=float("nan"))
            if not (math.isfinite(p0) and math.isfinite(p1)) or p0 <= 0 or p1 <= 0:
                continue
            fwd = (p1 - p0) / p0

            signals = row.get("signals", {})
            if not isinstance(signals, dict):
                continue

            realized_hist.append(fwd)
            for name, payload in signals.items():
                if not isinstance(payload, dict):
                    continue
                value = safe_float(payload.get("value", 0.0))
                signal_hist.setdefault(str(name), []).append(value)

    out: dict[str, float] = {}
    for name, values in signal_hist.items():
        out[name] = round(_pearson(values, realized_hist), 6)
    return out


def generate_performance_report(entries: list[dict[str, Any]], horizon: int = 1) -> dict[str, Any]:
    trades = build_trade_outcomes(entries, horizon=horizon)
    returns = [t.ret for t in trades]
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)

    trade_count = len(trades)
    accuracy = (wins / trade_count) if trade_count > 0 else 0.0
    avg_ret = (sum(returns) / trade_count) if trade_count > 0 else 0.0
    expectancy = avg_ret
    drawdown = _max_drawdown(returns) if returns else 0.0
    pf = _profit_factor(returns) if returns else 0.0
    avg_duration = (sum(t.duration_steps for t in trades) / trade_count) if trade_count > 0 else 0.0

    return {
        "trades": trade_count,
        "wins": wins,
        "losses": losses,
        "signal_accuracy": round(accuracy, 6),
        "win_rate": round(accuracy, 6),
        "profit_factor": pf if pf == float("inf") else round(pf, 6),
        "expectancy": round(expectancy, 6),
        "avg_trade_return": round(avg_ret, 6),
        "max_drawdown": round(drawdown, 6),
        "avg_trade_duration_steps": round(avg_duration, 4),
        "module_contributions": module_contribution_report(entries, horizon=horizon),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance analytics from trace logs")
    parser.add_argument("--trace", default="logs/recommendation_trace.jsonl", help="Path to recommendation trace jsonl")
    parser.add_argument("--horizon", type=int, default=1, help="Forward step horizon")
    args = parser.parse_args()

    rows = load_trace_entries(args.trace)
    report = generate_performance_report(rows, horizon=args.horizon)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
