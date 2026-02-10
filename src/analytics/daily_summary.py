from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.trace_utils import load_jsonl_dict_rows, safe_float


def _iso_day(ts: str) -> str:
    text = str(ts or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return text[:10]


def _next_step_returns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ticker = str(row.get("ticker", "")).upper().strip()
        if ticker:
            by_ticker[ticker].append(row)

    out: list[dict[str, Any]] = []
    for ticker, items in by_ticker.items():
        items.sort(key=lambda r: str(r.get("ts", "")))
        for i in range(len(items) - 1):
            a, b = items[i], items[i + 1]
            p0 = safe_float(a.get("price"), 0.0)
            p1 = safe_float(b.get("price"), 0.0)
            if p0 <= 0 or p1 <= 0:
                continue
            out.append(
                {
                    "ticker": ticker,
                    "decision": str(a.get("decision", "HOLD")).upper(),
                    "ret": (p1 - p0) / p0,
                }
            )
    return out


def generate_daily_summary(
    trace_path: str = "logs/recommendation_trace.jsonl",
    output_dir: str = "artifacts/reports",
    day: str | None = None,
) -> dict[str, Any]:
    rows = load_jsonl_dict_rows(trace_path)
    if day:
        rows = [r for r in rows if _iso_day(str(r.get("ts", ""))) == day]
    else:
        unique_days = sorted({_iso_day(str(r.get("ts", ""))) for r in rows if _iso_day(str(r.get("ts", "")))})
        if unique_days:
            day = unique_days[-1]
            rows = [r for r in rows if _iso_day(str(r.get("ts", ""))) == day]
        else:
            day = datetime.now(timezone.utc).date().isoformat()

    decision_counts = Counter()
    regime_counts = Counter()
    reason_counts = Counter()
    guardrail_counts = Counter()
    veto_counts = Counter()
    quality_total = 0
    quality_ok = 0
    confidence_values = []

    for row in rows:
        decision_counts[str(row.get("decision", "HOLD")).upper()] += 1
        risk_context = row.get("risk_context", {}) or {}
        regime_counts[str(risk_context.get("market_regime", "unknown"))] += 1
        confidence_values.append(safe_float(row.get("confidence", 0.0), 0.0))

        for reason in row.get("decision_reasons", []) or []:
            reason_text = str(reason)
            reason_counts[reason_text] += 1
            if reason_text.startswith("guardrail:"):
                guardrail_counts[reason_text] += 1
            if reason_text.startswith("veto:"):
                veto_counts[reason_text] += 1

        signals = row.get("signals", {}) or {}
        for payload in signals.values():
            if not isinstance(payload, dict):
                continue
            quality_total += 1
            if bool(payload.get("quality_ok", False)):
                quality_ok += 1

    realized = _next_step_returns(rows)
    pnl_by_decision = defaultdict(float)
    pnl_by_ticker = defaultdict(float)
    trades = 0
    wins = 0
    for rec in realized:
        decision = str(rec["decision"])
        if decision not in {"BUY", "SELL"}:
            continue
        ret = float(rec["ret"])
        signed = ret if decision == "BUY" else -ret
        pnl_by_decision[decision] += signed
        pnl_by_ticker[str(rec["ticker"])] += signed
        trades += 1
        if signed > 0:
            wins += 1

    summary = {
        "day": day,
        "rows": len(rows),
        "decision_counts": dict(decision_counts),
        "regime_distribution": dict(regime_counts),
        "signal_quality_ratio": (quality_ok / quality_total) if quality_total > 0 else 0.0,
        "avg_signal_confidence": (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0,
        "top_reasons": reason_counts.most_common(12),
        "guardrail_counts": dict(guardrail_counts),
        "veto_counts": dict(veto_counts),
        "pnl_attribution": {
            "by_decision": dict(pnl_by_decision),
            "by_ticker": dict(sorted(pnl_by_ticker.items(), key=lambda kv: kv[1], reverse=True)),
            "trade_count": trades,
            "win_rate": (wins / trades) if trades > 0 else 0.0,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_{day.replace('-', '')}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Flat csv for quick inspection.
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(
            [
                {"metric": "rows", "value": summary["rows"]},
                {"metric": "avg_signal_confidence", "value": summary["avg_signal_confidence"]},
                {"metric": "signal_quality_ratio", "value": summary["signal_quality_ratio"]},
                {"metric": "trade_count", "value": summary["pnl_attribution"]["trade_count"]},
                {"metric": "win_rate", "value": summary["pnl_attribution"]["win_rate"]},
            ]
        )

    summary["output_json"] = str(out_path)
    summary["output_csv"] = str(csv_path)
    return summary
