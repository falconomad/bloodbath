from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class GuardrailDecision:
    allow_execution: bool
    block_new_buys: bool
    reasons: list[str]
    filtered_analyses: list[dict[str, Any]]


def _parse_last_ts_from_price_df(data: Any) -> datetime | None:
    if data is None or getattr(data, "empty", True):
        return None
    try:
        idx = data.index
        if len(idx) <= 0:
            return None
        ts = idx[-1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)
    except Exception:
        return None
    return None


class ExecutionSafeguard:
    def __init__(
        self,
        state_path: str = "logs/execution_guard_state.json",
        stale_data_max_age_hours: float = 72.0,
        anomaly_zscore_threshold: float = 4.0,
        max_cycle_notional_turnover: float = 1.25,
        max_consecutive_failures: int = 3,
    ):
        self.state_path = Path(state_path)
        self.stale_data_max_age_hours = float(stale_data_max_age_hours)
        self.anomaly_zscore_threshold = float(anomaly_zscore_threshold)
        self.max_cycle_notional_turnover = float(max_cycle_notional_turnover)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"consecutive_failures": 0, "turnover_history": []}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {"consecutive_failures": 0, "turnover_history": []}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")

    def record_failure(self, reason: str) -> None:
        fails = int(self._state.get("consecutive_failures", 0))
        self._state["consecutive_failures"] = fails + 1
        self._state["last_failure_reason"] = str(reason)
        self._state["last_failure_ts"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def record_success(self) -> None:
        self._state["consecutive_failures"] = 0
        self._state.pop("last_failure_reason", None)
        self._state.pop("last_failure_ts", None)
        self._save_state()

    def _filter_buys(self, analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for row in analyses:
            payload = dict(row)
            if str(payload.get("decision", "HOLD")).upper() == "BUY":
                payload["decision"] = "HOLD"
                reasons = list(payload.get("decision_reasons", []) or [])
                reasons.append("guardrail:circuit_breaker")
                payload["decision_reasons"] = reasons
            out.append(payload)
        return out

    def _estimate_turnover(self, analyses: list[dict[str, Any]], portfolio_value: float) -> float:
        if portfolio_value <= 0:
            return 0.0
        total_size = 0.0
        for row in analyses:
            decision = str(row.get("decision", "HOLD")).upper()
            if decision not in {"BUY", "SELL"}:
                continue
            size = float(row.get("position_size", 0.0) or 0.0)
            if size <= 0:
                size = 0.1
            total_size += min(max(size, 0.0), 1.0)
        return total_size

    def _turnover_anomaly(self, turnover: float) -> bool:
        history = [float(x) for x in (self._state.get("turnover_history", []) or []) if math.isfinite(float(x))]
        if len(history) < 8:
            return False
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(max(variance, 0.0))
        if std <= 1e-9:
            return turnover > (mean * 2.5) and turnover > self.max_cycle_notional_turnover
        z = (turnover - mean) / std
        return z >= self.anomaly_zscore_threshold

    def _update_turnover_history(self, turnover: float) -> None:
        history = [float(x) for x in (self._state.get("turnover_history", []) or [])]
        history.append(float(turnover))
        self._state["turnover_history"] = history[-120:]
        self._save_state()

    def assess(
        self,
        analyses: list[dict[str, Any]],
        candidates: dict[str, dict[str, Any]],
        portfolio_value: float,
        now_utc: datetime | None = None,
    ) -> GuardrailDecision:
        current = now_utc or datetime.now(timezone.utc)
        reasons: list[str] = []

        fails = int(self._state.get("consecutive_failures", 0))
        if fails >= self.max_consecutive_failures:
            reasons.append(f"circuit_breaker:consecutive_failures={fails}")
            filtered = self._filter_buys(analyses)
            self._update_turnover_history(self._estimate_turnover(filtered, portfolio_value))
            return GuardrailDecision(False, True, reasons, filtered)

        stale = 0
        for ticker, meta in (candidates or {}).items():
            _ = ticker
            data = (meta or {}).get("data")
            last_ts = _parse_last_ts_from_price_df(data)
            if last_ts is None:
                continue
            age_hours = (current - last_ts).total_seconds() / 3600.0
            if age_hours > self.stale_data_max_age_hours:
                stale += 1
        if stale > 0:
            reasons.append(f"guardrail:stale_data={stale}")

        turnover = self._estimate_turnover(analyses, portfolio_value)
        if turnover > self.max_cycle_notional_turnover:
            reasons.append(f"guardrail:turnover_cap={turnover:.3f}")
        if self._turnover_anomaly(turnover):
            reasons.append(f"guardrail:turnover_anomaly={turnover:.3f}")

        block_buys = len(reasons) > 0
        filtered = self._filter_buys(analyses) if block_buys else analyses
        self._update_turnover_history(self._estimate_turnover(filtered, portfolio_value))
        return GuardrailDecision(not block_buys, block_buys, reasons, filtered)
