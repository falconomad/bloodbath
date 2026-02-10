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
            "volatility_low_atr_pct": 0.02,
            "volatility_high_atr_pct": 0.08,
            "volatility_confidence_penalty_max": 0.35,
            "volatility_position_scale_min": 0.35,
            "conflict_penalty": 0.20,
            "max_confidence_drop_on_conflict": 0.75,
            "max_data_gap_ratio_for_trade": 0.05,
            "require_micro_for_buy": True,
            "max_position_size": 0.20,
            "min_position_size": 0.02,
            "max_portfolio_drawdown_for_new_risk": 0.18,
            "max_sector_concentration": 0.45,
            "max_avg_correlation": 0.85,
            "portfolio_risk_confidence_penalty_max": 0.30,
            "portfolio_risk_position_scale_min": 0.40,
        },
        "regimes": {
            "bull": {"trend": 0.40, "sentiment": 0.24, "events": 0.12, "micro": 0.12, "dip": 0.08, "volatility": 0.04},
            "bear": {"trend": 0.32, "sentiment": 0.20, "events": 0.16, "micro": 0.10, "dip": 0.14, "volatility": 0.08},
            "volatile": {"trend": 0.30, "sentiment": 0.20, "events": 0.14, "micro": 0.08, "dip": 0.10, "volatility": 0.18},
            "sideways": {"trend": 0.30, "sentiment": 0.26, "events": 0.16, "micro": 0.12, "dip": 0.10, "volatility": 0.06},
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
        "growth": {
            "enabled": True,
            "min_growth_edge_for_trade": 0.22,
            "objective_total_return_weight": 1.0,
            "objective_expectancy_weight": 0.6,
            "objective_drawdown_penalty": 0.9,
            "objective_turnover_penalty": 0.15,
            "objective_risk_adjusted_weight": 0.2,
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


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    positive = {str(k): max(float(v), 0.0) for k, v in (weights or {}).items()}
    total = sum(positive.values())
    if total <= 0:
        n = max(len(positive), 1)
        return {k: 1.0 / n for k in positive} if positive else {}
    return {k: v / total for k, v in positive.items()}


def resolve_effective_weights(cfg: dict[str, Any], risk_context: dict[str, Any]) -> tuple[dict[str, float], str]:
    base = normalize_weights(cfg.get("weights", {}) or {})
    regime = str((risk_context or {}).get("market_regime", "default")).lower().strip()
    overrides = cfg.get("regimes", {}) or {}
    regime_weights = overrides.get(regime)
    if isinstance(regime_weights, dict) and regime_weights:
        return normalize_weights(regime_weights), regime
    return base, "default"


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

    max_drawdown = float(risk.get("max_portfolio_drawdown_for_new_risk", 0.18))
    portfolio_drawdown = float(risk_context.get("portfolio_drawdown", 0.0))
    if portfolio_drawdown > max_drawdown:
        reasons.append("guardrail:portfolio_drawdown")

    max_sector = float(risk.get("max_sector_concentration", 0.45))
    sector_alloc = float(risk_context.get("ticker_sector_allocation", 0.0))
    if sector_alloc > max_sector:
        reasons.append("guardrail:sector_concentration")

    max_corr = float(risk.get("max_avg_correlation", 0.85))
    avg_corr = float(risk_context.get("portfolio_avg_correlation", 0.0))
    if avg_corr > max_corr:
        reasons.append("guardrail:high_correlation")

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


def volatility_adjustment(risk_context: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float, str]:
    risk_cfg = cfg.get("risk", {})
    atr_pct = float(risk_context.get("atr_pct", 0.0))
    if atr_pct <= 0:
        return 1.0, 1.0, "unknown"

    low = float(risk_cfg.get("volatility_low_atr_pct", 0.02))
    high = max(float(risk_cfg.get("volatility_high_atr_pct", 0.08)), low + 1e-6)
    conf_penalty_max = clamp(float(risk_cfg.get("volatility_confidence_penalty_max", 0.35)), 0.0, 0.95)
    pos_scale_min = clamp(float(risk_cfg.get("volatility_position_scale_min", 0.35)), 0.0, 1.0)

    if atr_pct <= low:
        return 1.0, 1.0, "low"
    if atr_pct >= high:
        return 1.0 - conf_penalty_max, pos_scale_min, "high"

    t = (atr_pct - low) / (high - low)
    conf_mult = 1.0 - (conf_penalty_max * t)
    pos_mult = 1.0 - ((1.0 - pos_scale_min) * t)
    return clamp(conf_mult, 0.0, 1.0), clamp(pos_mult, 0.0, 1.0), "medium"


def portfolio_risk_adjustment(risk_context: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float, str]:
    risk_cfg = cfg.get("risk", {})
    drawdown = clamp(float(risk_context.get("portfolio_drawdown", 0.0)), 0.0, 1.0)
    sector_alloc = clamp(float(risk_context.get("ticker_sector_allocation", 0.0)), 0.0, 1.0)
    corr = clamp(float(risk_context.get("portfolio_avg_correlation", 0.0)), 0.0, 1.0)

    dd_cap = max(float(risk_cfg.get("max_portfolio_drawdown_for_new_risk", 0.18)), 1e-6)
    sec_cap = max(float(risk_cfg.get("max_sector_concentration", 0.45)), 1e-6)
    corr_cap = max(float(risk_cfg.get("max_avg_correlation", 0.85)), 1e-6)

    # Composite portfolio stress where >1 means above configured risk budgets.
    stress = max(drawdown / dd_cap, sector_alloc / sec_cap, corr / corr_cap)
    stress = clamp((stress - 1.0), 0.0, 1.0)
    if stress <= 0:
        return 1.0, 1.0, "normal"

    conf_penalty_max = clamp(float(risk_cfg.get("portfolio_risk_confidence_penalty_max", 0.30)), 0.0, 0.95)
    pos_scale_min = clamp(float(risk_cfg.get("portfolio_risk_position_scale_min", 0.40)), 0.0, 1.0)
    conf_mult = 1.0 - (conf_penalty_max * stress)
    pos_mult = 1.0 - ((1.0 - pos_scale_min) * stress)
    return clamp(conf_mult, 0.0, 1.0), clamp(pos_mult, 0.0, 1.0), "stressed"


def position_size(score: float, confidence: float, cfg: dict[str, Any], position_scale: float = 1.0) -> float:
    risk_cfg = cfg.get("risk", {})
    min_size = float(risk_cfg.get("min_position_size", 0.02))
    max_size = float(risk_cfg.get("max_position_size", 0.20))
    conviction = clamp(abs(score) * confidence, 0.0, 1.0)
    raw_size = min_size + (max_size - min_size) * conviction
    scaled = raw_size * clamp(position_scale, 0.0, 1.0)
    return round(clamp(scaled, 0.0, max_size), 4)


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

    conf_mult, pos_mult, vol_regime = volatility_adjustment(risk_context, cfg)
    p_conf_mult, p_pos_mult, portfolio_regime = portfolio_risk_adjustment(risk_context, cfg)
    adjusted_confidence = clamp(confidence * conf_mult * p_conf_mult, 0.0, 1.0)
    reasons.append(f"volatility:{vol_regime}")
    reasons.append(f"portfolio_risk:{portfolio_regime}")

    if adjusted_confidence < force_hold_conf:
        reasons.append("confidence:force_hold")
    else:
        if score >= buy_t:
            decision = "BUY"
        elif score <= sell_t:
            decision = "SELL"

    if adjusted_confidence < min_conf and decision != "HOLD":
        reasons.append("confidence:below_min")
        decision = "HOLD"

    sentiment_value = float(signals.get("sentiment", Signal("sentiment", 0.0, 0.0, False)).value)
    if decision == "BUY" and sentiment_value <= float(risk.get("extreme_negative_sentiment", -0.7)):
        reasons.append("risk:extreme_negative_sentiment")
        decision = "HOLD"

    if local_conflict >= conflict_hold_t:
        reasons.append("conflict:high_disagreement")
        decision = "HOLD"

    growth_cfg = cfg.get("growth", {})
    if bool(growth_cfg.get("enabled", True)) and decision in {"BUY", "SELL"}:
        growth_edge = abs(float(score)) * adjusted_confidence
        min_growth_edge = clamp(float(growth_cfg.get("min_growth_edge_for_trade", 0.22)), 0.0, 1.0)
        if growth_edge < min_growth_edge:
            reasons.append("growth:edge_below_min")
            decision = "HOLD"

    if guardrail_reasons:
        reasons.extend(guardrail_reasons)

    vetoed_decision, veto_reasons = veto_decision(
        proposed_decision=decision,
        confidence=adjusted_confidence,
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

    size = 0.0 if decision != "BUY" else position_size(score, adjusted_confidence, cfg, position_scale=(pos_mult * p_pos_mult))
    return decision, reasons, size


def write_trace(payload: dict[str, Any]) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    with TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
