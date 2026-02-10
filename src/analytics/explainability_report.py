from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

import json

from src.common.trace_utils import load_jsonl_dict_rows


def load_trace_entries(trace_path: str) -> list[dict[str, Any]]:
    return load_jsonl_dict_rows(trace_path)


def _top_signal_contributors(signals: dict[str, Any], weights: dict[str, Any], limit: int = 3) -> list[dict[str, float]]:
    parts = []
    for name, payload in (signals or {}).items():
        if not isinstance(payload, dict):
            continue
        value = float(payload.get("value", 0.0))
        w = float(weights.get(name, 0.0)) if isinstance(weights, dict) else 0.0
        parts.append({"signal": name, "contribution": value * w, "value": value, "weight": w})
    parts.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return parts[:limit]


def generate_explainability_report(entries: list[dict[str, Any]], max_examples: int = 20) -> dict[str, Any]:
    decision_counts = Counter()
    reason_counts = Counter()
    veto_counts = Counter()
    guardrail_counts = Counter()
    per_ticker = defaultdict(lambda: Counter())
    examples = []

    for row in entries:
        ticker = str(row.get("ticker", "")).upper().strip() or "UNKNOWN"
        decision = str(row.get("decision", "HOLD")).upper().strip()
        reasons = [str(r) for r in (row.get("decision_reasons", []) or [])]
        decision_counts[decision] += 1
        per_ticker[ticker][decision] += 1

        for r in reasons:
            reason_counts[r] += 1
            if r.startswith("veto:"):
                veto_counts[r] += 1
            if r.startswith("guardrail:"):
                guardrail_counts[r] += 1

        if len(examples) < max_examples:
            examples.append(
                {
                    "ts": row.get("ts"),
                    "ticker": ticker,
                    "decision": decision,
                    "score": row.get("score"),
                    "confidence": row.get("confidence"),
                    "reasons": reasons,
                    "top_contributors": _top_signal_contributors(row.get("signals", {}), row.get("weights", {})),
                }
            )

    return {
        "total_entries": len(entries),
        "decision_counts": dict(decision_counts),
        "top_reasons": reason_counts.most_common(15),
        "top_vetoes": veto_counts.most_common(10),
        "top_guardrails": guardrail_counts.most_common(10),
        "ticker_decision_summary": {k: dict(v) for k, v in per_ticker.items()},
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate human-readable explainability report from trace logs")
    parser.add_argument("--trace", default="logs/recommendation_trace.jsonl")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    rows = load_trace_entries(args.trace)
    report = generate_explainability_report(rows, max_examples=args.max_examples)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
