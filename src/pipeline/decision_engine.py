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
            "high_conflict_force_hold": 0.7,
        },
        "quality": {
            "min_price_points": 40,
            "max_missing_ratio": 0.05,
            "min_news_articles": 3,
            "min_sentiment_articles": 4,
            "min_usable_signals": 3,
        },
        "risk": {
            "min_rel_volume_for_buy": 0.75,
            "extreme_negative_sentiment": -0.7,
            "max_atr_pct_for_full_risk": 0.08,
            "conflict_penalty": 0.20,
            "max_confidence_drop_on_conflict": 0.75,
            "max_data_gap_ratio_for_trade": 0.05,
            "require_micro_for_buy": True,
            "max_position_size": 0.20,
            "min_position_size": 0.02,
        },
        "stability": {
            "min_decision_hold_cycles": 2,
            "min_cycles_between_flips": 2,
            "min_cycles_between_non_hold_signals": 1,
        },
        "veto": {
            "enabled": True,
            "block_on_any_guardrail": True,
            "max_conflict_for_trade": 0.7,
            "require_trend_alignment": True,
            "min_quality_signals_for_trade": 3,
            "min_confidence_for_trade": 0.45,
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


def aggregate_confidence(
    signals: dict[str, Signal],
    conflict_penalty: float,
    weights: dict[str, float] | None = None,
    max_conflict_drop: float = 0.75,
) -> tuple[float, float]:
    usable = [(name, s) for name, s in signals.items() if s.quality_ok]
    if not usable:
        return 0.0, 1.0

    if weights:
        weighted_conf = 0.0
        total_w = 0.0
        for name, signal in usable:
            w = max(float(weights.get(name, 0.0)), 0.0)
            if w <= 0:
                continue
            weighted_conf += signal.confidence * w
            total_w += w
        avg_conf = (weighted_conf / total_w) if total_w > 0 else (sum(s.confidence for _, s in usable) / len(usable))
    else:
        avg_conf = sum(s.confidence for _, s in usable) / len(usable)

    conflicts = conflict_ratio(signals)
    # Conflict grows non-linearly so direct opposition drops confidence sharply.
    penalty = (conflicts**2) * float(conflict_penalty)
    penalty = clamp(penalty, 0.0, float(max_conflict_drop))
    final_conf = clamp(avg_conf * (1.0 - penalty), 0.0, 1.0)
    return final_conf, conflicts


def hard_guardrails(
    signals: dict[str, Signal],
    risk_context: dict[str, Any],
    cfg: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    quality = cfg.get("quality", {})
    risk = cfg.get("risk", {})

    usable_signals = sum(1 for s in signals.values() if s.quality_ok)
    if usable_signals < int(quality.get("min_usable_signals", 3)):
        reasons.append("guardrail:insufficient_usable_signals")

    data_quality_ok = bool(risk_context.get("data_quality_ok", True))
    if not data_quality_ok:
        reasons.append("guardrail:price_data_quality")

    data_gap_ratio = float(risk_context.get("data_gap_ratio", 0.0))
    if data_gap_ratio > float(risk.get("max_data_gap_ratio_for_trade", 0.05)):
        reasons.append("guardrail:data_gaps")

    min_rel_volume = float(risk.get("min_rel_volume_for_buy", 0.75))
    rel_volume = float(risk_context.get("rel_volume", 1.0))
    if rel_volume < min_rel_volume:
        reasons.append("guardrail:low_volume")

    require_micro_for_buy = bool(risk.get("require_micro_for_buy", True))
    if require_micro_for_buy and not bool(risk_context.get("micro_available", False)):
        reasons.append("guardrail:micro_unavailable")

    return reasons


def veto_decision(
    proposed_decision: str,
    confidence: float,
    signals: dict[str, Signal],
    guardrail_reasons: list[str],
    local_conflict: float,
    cfg: dict[str, Any],
) -> tuple[str, list[str]]:
    veto_cfg = cfg.get("veto", {})
    if not bool(veto_cfg.get("enabled", True)):
        return proposed_decision, []

    vetoes: list[str] = []
    if proposed_decision == "HOLD":
        return "HOLD", vetoes

    if bool(veto_cfg.get("block_on_any_guardrail", True)) and guardrail_reasons:
        vetoes.append("veto:guardrail")

    max_conflict_for_trade = float(veto_cfg.get("max_conflict_for_trade", 0.7))
    if local_conflict >= max_conflict_for_trade:
        vetoes.append("veto:high_conflict")

    min_conf_trade = float(veto_cfg.get("min_confidence_for_trade", 0.45))
    if confidence < min_conf_trade:
        vetoes.append("veto:low_confidence")

    min_quality_signals = int(veto_cfg.get("min_quality_signals_for_trade", 3))
    usable_signals = sum(1 for s in signals.values() if s.quality_ok)
    if usable_signals < min_quality_signals:
        vetoes.append("veto:insufficient_signal_quality")

    if bool(veto_cfg.get("require_trend_alignment", True)):
        trend_value = float(signals.get("trend", Signal("trend", 0.0, 0.0, False)).value)
        if proposed_decision == "BUY" and trend_value <= 0:
            vetoes.append("veto:trend_misaligned")
        if proposed_decision == "SELL" and trend_value >= 0:
            vetoes.append("veto:trend_misaligned")

    risk_cfg = cfg.get("risk", {})
    sentiment_value = float(signals.get("sentiment", Signal("sentiment", 0.0, 0.0, False)).value)
    if proposed_decision == "BUY" and sentiment_value <= float(risk_cfg.get("extreme_negative_sentiment", -0.7)):
        vetoes.append("veto:extreme_negative_sentiment")

    if vetoes:
        return "HOLD", vetoes
    return proposed_decision, vetoes


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
    min_non_hold_gap = int(stability.get("min_cycles_between_non_hold_signals", 1))

    previous = state.get(ticker)
    if not previous:
        state[ticker] = {
            "decision": proposed_decision,
            "cycle": cycle_idx,
            "flip_cycle": cycle_idx,
            "last_non_hold_cycle": cycle_idx if proposed_decision != "HOLD" else -10_000,
        }
        return proposed_decision, ""

    prev_decision = str(previous.get("decision", "HOLD"))
    prev_cycle = int(previous.get("cycle", cycle_idx))
    prev_flip = int(previous.get("flip_cycle", prev_cycle))
    prev_non_hold = int(previous.get("last_non_hold_cycle", prev_cycle))

    if proposed_decision == prev_decision:
        previous["cycle"] = cycle_idx
        if proposed_decision != "HOLD":
            previous["last_non_hold_cycle"] = cycle_idx
        return proposed_decision, ""

    if cycle_idx - prev_cycle < min_hold:
        return prev_decision, "stability:min_hold"

    if cycle_idx - prev_flip < min_flip_gap:
        return prev_decision, "stability:min_flip_gap"

    if proposed_decision != "HOLD" and (cycle_idx - prev_non_hold < min_non_hold_gap):
        return "HOLD", "stability:min_non_hold_gap"

    previous.update({"decision": proposed_decision, "cycle": cycle_idx, "flip_cycle": cycle_idx})
    if proposed_decision != "HOLD":
        previous["last_non_hold_cycle"] = cycle_idx
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
    conflict_hold_t = float(thresholds.get("high_conflict_force_hold", 0.7))

    guardrail_reasons = hard_guardrails(signals, risk_context, cfg)

    local_conflict = conflict_ratio(signals)

    if confidence < force_hold_conf:
        reasons.append("confidence:force_hold")
    else:
        if score >= buy_t:
            decision = "BUY"
        elif score <= sell_t:
            decision = "SELL"

    if confidence < min_conf and decision != "HOLD":
        reasons.append("confidence:below_min")

    sentiment_value = float(signals.get("sentiment", Signal("sentiment", 0.0, 0.0, False)).value)
    if decision == "BUY" and sentiment_value <= float(risk.get("extreme_negative_sentiment", -0.7)):
        reasons.append("risk:extreme_negative_sentiment")

    if local_conflict >= conflict_hold_t:
        reasons.append("conflict:high_disagreement")

    if guardrail_reasons:
        reasons.extend(guardrail_reasons)

    vetoed_decision, veto_reasons = veto_decision(
        proposed_decision=decision,
        confidence=confidence,
        signals=signals,
        guardrail_reasons=guardrail_reasons,
        local_conflict=local_conflict,
        cfg=cfg,
    )
    if veto_reasons:
        reasons.extend(veto_reasons)
    decision = vetoed_decision

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
