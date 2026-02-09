from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


WEIGHTS_PATH = Path("config/weights.yaml")
TRACE_PATH = Path("logs/recommendation_trace.jsonl")


@dataclass
class Signal:
    name: str
    value: float
    confidence: float
    quality_ok: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "confidence": round(self.confidence, 6),
            "quality_ok": bool(self.quality_ok),
            "reason": self.reason,
        }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _fallback_config() -> dict[str, Any]:
    return {
        "weights": {
            "trend": 0.35,
            "sentiment": 0.25,
            "events": 0.15,
            "micro": 0.10,
            "dip": 0.10,
            "volatility": 0.05,
        },
        "thresholds": {
            "buy_score": 0.45,
            "sell_score": -0.45,
            "min_confidence": 0.45,
            "force_hold_confidence": 0.35,
        },
        "quality": {
            "min_price_points": 40,
            "max_missing_ratio": 0.05,
            "min_news_articles": 3,
            "min_sentiment_articles": 4,
        },
        "risk": {
            "min_rel_volume_for_buy": 0.75,
            "extreme_negative_sentiment": -0.7,
            "max_atr_pct_for_full_risk": 0.08,
            "conflict_penalty": 0.20,
            "max_position_size": 0.20,
            "min_position_size": 0.02,
        },
        "stability": {
            "min_decision_hold_cycles": 2,
            "min_cycles_between_flips": 2,
        },
    }


def load_config() -> dict[str, Any]:
    if yaml is None or not WEIGHTS_PATH.exists():
        return _fallback_config()
    try:
        with WEIGHTS_PATH.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception:
        return _fallback_config()

    cfg = _fallback_config()
    for key in cfg:
        if isinstance(loaded.get(key), dict):
            cfg[key].update(loaded[key])
    return cfg


def normalize_signals(signals: dict[str, Signal]) -> dict[str, Signal]:
    for signal in signals.values():
        signal.value = clamp(signal.value, -1.0, 1.0)
        signal.confidence = clamp(signal.confidence, 0.0, 1.0)
    return signals


def weighted_score(signals: dict[str, Signal], weights: dict[str, float]) -> float:
    active = []
    for name, signal in signals.items():
        if not signal.quality_ok:
            continue
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        active.append((signal, w))

    if not active:
        return 0.0

    numerator = sum(s.value * w for s, w in active)
    denominator = sum(w for _, w in active)
    if denominator <= 0:
        return 0.0
    return clamp(numerator / denominator, -1.0, 1.0)


def conflict_ratio(signals: dict[str, Signal]) -> float:
    votes = []
    for signal in signals.values():
        if not signal.quality_ok:
            continue
        if abs(signal.value) < 0.1:
            continue
        votes.append(1 if signal.value > 0 else -1)
    if len(votes) <= 1:
        return 0.0
    aligned = abs(sum(votes)) / len(votes)
    return clamp(1.0 - aligned, 0.0, 1.0)


def aggregate_confidence(signals: dict[str, Signal], conflict_penalty: float) -> tuple[float, float]:
    usable = [s for s in signals.values() if s.quality_ok]
    if not usable:
        return 0.0, 1.0

    avg_conf = sum(s.confidence for s in usable) / len(usable)
    conflicts = conflict_ratio(signals)
    penalty = conflicts * float(conflict_penalty)
    final_conf = clamp(avg_conf * (1.0 - penalty), 0.0, 1.0)
    return final_conf, conflicts


def position_size(score: float, confidence: float, cfg: dict[str, Any]) -> float:
    risk_cfg = cfg.get("risk", {})
    min_size = float(risk_cfg.get("min_position_size", 0.02))
    max_size = float(risk_cfg.get("max_position_size", 0.20))
    conviction = clamp(abs(score) * confidence, 0.0, 1.0)
    return round(min_size + (max_size - min_size) * conviction, 4)


def apply_stability(
    ticker: str,
    proposed_decision: str,
    state: dict[str, dict[str, Any]],
    cycle_idx: int,
    cfg: dict[str, Any],
) -> tuple[str, str]:
    stability = cfg.get("stability", {})
    min_hold = int(stability.get("min_decision_hold_cycles", 2))
    min_flip_gap = int(stability.get("min_cycles_between_flips", 2))

    previous = state.get(ticker)
    if not previous:
        state[ticker] = {"decision": proposed_decision, "cycle": cycle_idx, "flip_cycle": cycle_idx}
        return proposed_decision, ""

    prev_decision = str(previous.get("decision", "HOLD"))
    prev_cycle = int(previous.get("cycle", cycle_idx))
    prev_flip = int(previous.get("flip_cycle", prev_cycle))

    if proposed_decision == prev_decision:
        previous["cycle"] = cycle_idx
        return proposed_decision, ""

    if cycle_idx - prev_cycle < min_hold:
        return prev_decision, "stability:min_hold"

    if cycle_idx - prev_flip < min_flip_gap:
        return prev_decision, "stability:min_flip_gap"

    previous.update({"decision": proposed_decision, "cycle": cycle_idx, "flip_cycle": cycle_idx})
    return proposed_decision, ""


def decide(
    ticker: str,
    score: float,
    confidence: float,
    signals: dict[str, Signal],
    risk_context: dict[str, Any],
    state: dict[str, dict[str, Any]],
    cycle_idx: int,
    cfg: dict[str, Any],
) -> tuple[str, list[str], float]:
    thresholds = cfg.get("thresholds", {})
    risk = cfg.get("risk", {})

    reasons: list[str] = []
    decision = "HOLD"

    force_hold_conf = float(thresholds.get("force_hold_confidence", 0.35))
    min_conf = float(thresholds.get("min_confidence", 0.45))
    buy_t = float(thresholds.get("buy_score", 0.45))
    sell_t = float(thresholds.get("sell_score", -0.45))

    if confidence < force_hold_conf:
        reasons.append("confidence:force_hold")
    else:
        if score >= buy_t:
            decision = "BUY"
        elif score <= sell_t:
            decision = "SELL"

    if confidence < min_conf and decision != "HOLD":
        decision = "HOLD"
        reasons.append("confidence:below_min")

    sentiment_value = float(signals.get("sentiment", Signal("sentiment", 0.0, 0.0, False)).value)
    rel_volume = float(risk_context.get("rel_volume", 1.0))
    if decision == "BUY" and sentiment_value <= float(risk.get("extreme_negative_sentiment", -0.7)):
        decision = "HOLD"
        reasons.append("risk:extreme_negative_sentiment")
    if decision == "BUY" and rel_volume < float(risk.get("min_rel_volume_for_buy", 0.75)):
        decision = "HOLD"
        reasons.append("risk:low_volume")

    decision, stability_reason = apply_stability(ticker, decision, state, cycle_idx, cfg)
    if stability_reason:
        reasons.append(stability_reason)

    size = 0.0 if decision != "BUY" else position_size(score, confidence, cfg)
    return decision, reasons, size


def write_trace(payload: dict[str, Any]) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    with TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
