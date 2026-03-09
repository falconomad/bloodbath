from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")


def log_event(event_type: str, payload: dict[str, Any], path: str = "logs/engine_events.jsonl") -> None:
    _append_jsonl(
        path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        },
    )


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def update_profit_summary(
    account_snapshot: dict[str, Any],
    positions_snapshot: list[dict[str, Any]],
    trades_executed: int,
    run_status: str,
    path: str = "logs/profit_summary.json",
    baseline_equity_hint: float = 0.0,
) -> dict[str, Any]:
    state = _read_json(path)
    now = datetime.now(timezone.utc).isoformat()

    current_equity = float(account_snapshot.get("equity", 0.0) or 0.0)
    last_equity = float(account_snapshot.get("last_equity", current_equity) or current_equity)
    daily_profit = current_equity - last_equity
    daily_profit_pct = (daily_profit / last_equity) * 100.0 if last_equity > 0 else 0.0

    baseline_equity = float(state.get("baseline_equity", 0.0) or 0.0)
    if baseline_equity_hint and baseline_equity_hint > 0:
        baseline_equity = float(baseline_equity_hint)
    elif baseline_equity <= 0:
        baseline_equity = current_equity

    total_profit = current_equity - baseline_equity
    total_return_pct = (total_profit / baseline_equity) * 100.0 if baseline_equity > 0 else 0.0

    unrealized_pl = 0.0
    for p in positions_snapshot or []:
        if p.get("unrealized_pl") is not None:
            try:
                unrealized_pl += float(p.get("unrealized_pl", 0.0) or 0.0)
                continue
            except Exception:
                pass
        try:
            mv = float(p.get("market_value", 0.0) or 0.0)
            pnl_pct = float(p.get("unrealized_pl_pct", 0.0) or 0.0) / 100.0
            cost_basis = mv / (1.0 + pnl_pct) if (1.0 + pnl_pct) != 0 else mv
            unrealized_pl += (mv - cost_basis)
        except Exception:
            continue

    run_count = int(state.get("run_count", 0) or 0) + 1
    cumulative_executed_orders = int(state.get("cumulative_executed_orders", 0) or 0) + int(trades_executed or 0)

    summary = {
        "last_updated_utc": now,
        "run_count": run_count,
        "run_status": str(run_status),
        "equity": round(current_equity, 2),
        "last_equity": round(last_equity, 2),
        "daily_profit": round(daily_profit, 2),
        "daily_profit_pct": round(daily_profit_pct, 4),
        "baseline_equity": round(baseline_equity, 2),
        "total_profit_since_baseline": round(total_profit, 2),
        "total_return_since_baseline_pct": round(total_return_pct, 4),
        "unrealized_pl_open_positions": round(unrealized_pl, 2),
        "open_positions_count": len(positions_snapshot or []),
        "trades_executed_this_run": int(trades_executed or 0),
        "cumulative_executed_orders": cumulative_executed_orders,
    }
    _write_json(path, summary)
    return summary
