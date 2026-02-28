from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.agent.goal_policy import GoalSnapshot


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def apply_goal_policy_to_config(cfg: dict[str, Any], goal: GoalSnapshot | None) -> dict[str, Any]:
    """Adjust thresholds/sizing from goal pacing without mutating base cfg."""
    effective = deepcopy(cfg)
    if goal is None:
        return effective

    thresholds = effective.setdefault("thresholds", {})
    risk = effective.setdefault("risk", {})

    pace = float(goal.pace_multiplier)
    # Behind pace => slightly lower buy threshold and allow a bit more sizing.
    buy_score = float(thresholds.get("buy_score", 0.45))
    min_conf = float(thresholds.get("min_confidence", 0.45))
    max_pos = float(risk.get("max_position_size", 0.20))

    thresholds["buy_score"] = round(_clamp(buy_score - ((pace - 1.0) * 0.12), 0.30, 0.70), 4)
    thresholds["min_confidence"] = round(_clamp(min_conf - ((pace - 1.0) * 0.08), 0.30, 0.70), 4)
    risk["max_position_size"] = round(_clamp(max_pos + ((pace - 1.0) * 0.10), 0.10, 0.30), 4)

    # Near deadline and still below target: increase edge requirement to avoid random churn.
    growth = effective.setdefault("growth", {})
    if goal.days_remaining <= 1.5 and goal.remaining_capital > 0:
        growth["min_growth_edge_for_trade"] = round(
            _clamp(float(growth.get("min_growth_edge_for_trade", 0.22)) + 0.05, 0.20, 0.45),
            4,
        )
    return effective
