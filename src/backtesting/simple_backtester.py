from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.decision_engine import Signal, aggregate_confidence, decide, load_config, normalize_signals, weighted_score


@dataclass
class BacktestExample:
    ticker: str
    ts: str
    price: float
    forward_return: float
    signals: dict[str, Signal]
    risk_context: dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def load_trace_entries(trace_path: str) -> list[dict[str, Any]]:
    path = Path(trace_path)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            rows.append(payload)
    return rows


def _signals_from_payload(raw_signals: dict[str, Any]) -> dict[str, Signal]:
    parsed: dict[str, Signal] = {}
    for name, payload in (raw_signals or {}).items():
        if not isinstance(payload, dict):
            continue
        parsed[str(name)] = Signal(
            name=str(name),
            value=_safe_float(payload.get("value", 0.0)),
            confidence=_safe_float(payload.get("confidence", 0.0)),
            quality_ok=bool(payload.get("quality_ok", False)),
            reason=str(payload.get("reason", "")),
        )
    return normalize_signals(parsed)


def build_examples(entries: list[dict[str, Any]], horizon: int = 1) -> list[BacktestExample]:
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in entries:
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        price = _safe_float(row.get("price", None), default=float("nan"))
        if not math.isfinite(price) or price <= 0:
            continue
        by_ticker.setdefault(ticker, []).append(row)

    examples: list[BacktestExample] = []
    for ticker, rows in by_ticker.items():
        rows.sort(key=lambda r: str(r.get("ts", "")))
        for idx, row in enumerate(rows):
            next_idx = idx + max(int(horizon), 1)
            if next_idx >= len(rows):
                continue
            next_price = _safe_float(rows[next_idx].get("price", 0.0))
            curr_price = _safe_float(row.get("price", 0.0))
            if curr_price <= 0 or next_price <= 0:
                continue
            forward_return = (next_price - curr_price) / curr_price
            signals = _signals_from_payload(row.get("signals", {}))
            if not signals:
                continue
            examples.append(
                BacktestExample(
                    ticker=ticker,
                    ts=str(row.get("ts", "")),
                    price=curr_price,
                    forward_return=float(forward_return),
                    signals=signals,
                    risk_context=dict(row.get("risk_context", {}) or {}),
                )
            )

    examples.sort(key=lambda e: e.ts)
    return examples


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    positive = {k: max(_safe_float(v), 0.0) for k, v in weights.items()}
    total = sum(positive.values())
    if total <= 0:
        n = max(len(positive), 1)
        return {k: 1.0 / n for k in positive} if positive else {}
    return {k: v / total for k, v in positive.items()}


def sample_weight_candidates(base_weights: dict[str, float], trials: int, seed: int = 7) -> list[dict[str, float]]:
    rng = random.Random(seed)
    keys = list(base_weights.keys())
    candidates = [_normalize_weights(base_weights)]
    for _ in range(max(trials - 1, 0)):
        perturbed = {}
        for k in keys:
            scale = rng.uniform(0.5, 1.5)
            perturbed[k] = max(0.0001, _safe_float(base_weights.get(k, 0.0)) * scale)
        candidates.append(_normalize_weights(perturbed))
    return candidates


def evaluate_candidate(examples: list[BacktestExample], cfg: dict[str, Any], weights: dict[str, float]) -> dict[str, float]:
    local_cfg = copy.deepcopy(cfg)
    local_cfg.setdefault("weights", {})
    local_cfg["weights"] = _normalize_weights(weights)

    pnl: list[float] = []
    trades = 0
    wins = 0
    state: dict[str, dict[str, Any]] = {}

    for i, ex in enumerate(examples, start=1):
        score = weighted_score(ex.signals, local_cfg.get("weights", {}))
        conf, _ = aggregate_confidence(
            ex.signals,
            local_cfg.get("risk", {}).get("conflict_penalty", 0.2),
            weights=local_cfg.get("weights", {}),
            max_conflict_drop=local_cfg.get("risk", {}).get("max_confidence_drop_on_conflict", 0.75),
        )
        decision, _reasons, _size = decide(
            ticker=ex.ticker,
            score=score,
            confidence=conf,
            signals=ex.signals,
            risk_context=ex.risk_context,
            state=state,
            cycle_idx=i,
            cfg=local_cfg,
        )

        trade_ret = 0.0
        if decision == "BUY":
            trade_ret = ex.forward_return
            trades += 1
        elif decision == "SELL":
            trade_ret = -ex.forward_return
            trades += 1

        if decision in {"BUY", "SELL"} and trade_ret > 0:
            wins += 1

        pnl.append(trade_ret)

    if not pnl:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "total_return": 0.0,
            "objective": -1e9,
        }

    total_return = sum(pnl)
    avg = total_return / len(pnl)
    variance = sum((x - avg) ** 2 for x in pnl) / len(pnl)
    std = math.sqrt(max(variance, 0.0))
    objective = (avg / std) if std > 1e-9 else (avg if avg > 0 else -1e6)

    return {
        "trades": float(trades),
        "win_rate": (wins / trades) if trades > 0 else 0.0,
        "avg_trade_return": (sum(pnl) / trades) if trades > 0 else 0.0,
        "total_return": total_return,
        "objective": objective,
    }


def tune_from_trace(trace_path: str, trials: int = 100, horizon: int = 1, seed: int = 7) -> dict[str, Any]:
    cfg = load_config()
    entries = load_trace_entries(trace_path)
    examples = build_examples(entries, horizon=horizon)
    if not examples:
        return {"status": "no_data", "examples": 0}

    base_weights = _normalize_weights(cfg.get("weights", {}))
    candidates = sample_weight_candidates(base_weights, trials=trials, seed=seed)

    best_weights = base_weights
    best_metrics = evaluate_candidate(examples, cfg, base_weights)

    for w in candidates[1:]:
        metrics = evaluate_candidate(examples, cfg, w)
        if metrics["objective"] > best_metrics["objective"]:
            best_metrics = metrics
            best_weights = w

    return {
        "status": "ok",
        "examples": len(examples),
        "best_weights": best_weights,
        "base_weights": base_weights,
        "best_metrics": best_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple trace-based weight tuner")
    parser.add_argument("--trace", default="logs/recommendation_trace.jsonl", help="Path to trace jsonl")
    parser.add_argument("--trials", type=int, default=100, help="Number of candidate weight sets")
    parser.add_argument("--horizon", type=int, default=1, help="Forward steps for return label")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    result = tune_from_trace(args.trace, trials=args.trials, horizon=args.horizon, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
