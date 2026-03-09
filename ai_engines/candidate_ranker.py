from __future__ import annotations

from statistics import mean
from typing import Any


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return ((a - b) / b) * 100.0


def detect_market_regime(spy_history: list[dict[str, Any]]) -> str:
    if not spy_history or len(spy_history) < 3:
        return "neutral"
    closes = [float(x.get("c", 0.0)) for x in spy_history if float(x.get("c", 0.0)) > 0]
    if len(closes) < 3:
        return "neutral"
    ret_3d = _pct(closes[-1], closes[-3])
    ret_1d = _pct(closes[-1], closes[-2])
    if ret_3d > 1.5 and ret_1d >= -0.5:
        return "risk_on"
    if ret_3d < -1.5 or ret_1d < -1.2:
        return "risk_off"
    return "neutral"


def build_features(symbol: str, bars: list[dict[str, Any]], spy_history: list[dict[str, Any]], news_items: list[dict[str, Any]]):
    # Daily bars can be very sparse around weekends/holidays/market-open timing.
    # Accept thin history and degrade feature quality gracefully instead of rejecting all symbols.
    if not bars:
        return None
    closes = [float(x.get("c", 0.0)) for x in bars if float(x.get("c", 0.0)) > 0]
    vols = [float(x.get("v", 0.0)) for x in bars if float(x.get("v", 0.0)) >= 0]
    if len(closes) < 1:
        return None

    ret_1d = _pct(closes[-1], closes[-2]) if len(closes) >= 2 else 0.0
    ret_3d = _pct(closes[-1], closes[-4]) if len(closes) >= 4 else (_pct(closes[-1], closes[0]) if len(closes) >= 2 else 0.0)
    ret_5d = _pct(closes[-1], closes[0])

    ema3 = mean(closes[-3:]) if len(closes) >= 3 else mean(closes)
    ema5 = mean(closes[-5:]) if len(closes) >= 5 else mean(closes)
    trend_gap = _pct(ema3, ema5)

    vol_window = vols[-5:] if len(vols) >= 5 else vols
    avg_vol = mean(vol_window) if vol_window else 0.0
    vol_mult = (vols[-1] / avg_vol) if avg_vol > 0 else 1.0
    vol_stability = 1.0
    if len(vols) >= 2 and avg_vol > 0:
        dev = mean(abs(v - avg_vol) / avg_vol for v in vol_window)
        vol_stability = _clamp(1.0 - dev, 0.0, 1.0)

    spy_closes = [float(x.get("c", 0.0)) for x in spy_history if float(x.get("c", 0.0)) > 0]
    spy_ret_5d = _pct(spy_closes[-1], spy_closes[0]) if len(spy_closes) >= 2 else 0.0
    rel_strength_5d = ret_5d - spy_ret_5d

    news_count = len(news_items or [])
    news_density = _clamp(news_count / 3.0, 0.0, 1.0)
    source_diversity = len({str(x.get("source", "")).lower() for x in (news_items or []) if str(x.get("source", "")).strip()})
    source_diversity = _clamp(source_diversity / 3.0, 0.0, 1.0)

    return {
        "symbol": symbol,
        "history_points": len(closes),
        "ret_1d": ret_1d,
        "ret_3d": ret_3d,
        "ret_5d": ret_5d,
        "trend_gap": trend_gap,
        "vol_mult": vol_mult,
        "vol_stability": vol_stability,
        "rel_strength_5d": rel_strength_5d,
        "news_count": news_count,
        "news_density": news_density,
        "source_diversity": source_diversity,
        "last_close": closes[-1],
    }


def hard_reject_reasons(features: dict[str, float], regime: str):
    reasons = []
    if features["last_close"] < 5.0:
        reasons.append("price_too_low")
    if features["vol_mult"] < 0.45:
        reasons.append("volume_too_thin")
    if features["news_count"] == 0:
        reasons.append("no_news_context")
    if regime == "risk_off" and features["ret_1d"] < -2.0:
        reasons.append("risk_off_momentum_break")
    return reasons


def rank_candidate(features: dict[str, float], tech_score: int, sent_score: int, regime: str):
    # Normalize local features to 0-100 components.
    momentum = 50 + (features["ret_3d"] * 6.0) + (features["ret_1d"] * 4.0)
    momentum = _clamp(momentum, 0, 100)

    rel = 50 + (features["rel_strength_5d"] * 5.0)
    rel = _clamp(rel, 0, 100)

    liquidity = _clamp((features["vol_mult"] * 35.0) + (features["vol_stability"] * 65.0), 0, 100)
    context = _clamp(((0.75 * features["news_density"]) + (0.25 * features.get("source_diversity", 0.0))) * 100.0, 0, 100)

    regime_bias = 0.0
    if regime == "risk_on":
        regime_bias = 5.0
    elif regime == "risk_off":
        regime_bias = -10.0

    local_score = (
        0.25 * momentum
        + 0.20 * rel
        + 0.20 * liquidity
        + 0.15 * context
        + 0.12 * tech_score
        + 0.08 * sent_score
        + regime_bias
    )
    local_score = _clamp(local_score, 0, 100)

    # Confidence penalizes disagreement and weak context.
    disagreement = abs(float(tech_score) - float(sent_score)) / 100.0
    confidence = (
        0.43
        + (0.28 * (min(tech_score, sent_score) / 100.0))
        + (0.18 * features["vol_stability"])
        + (0.13 * features["news_density"])
        + (0.08 * features.get("source_diversity", 0.0))
        - (0.25 * disagreement)
    )
    if regime == "risk_off":
        confidence -= 0.08
    confidence = _clamp(confidence, 0.0, 1.0)

    return {
        "local_score": round(local_score, 2),
        "confidence": round(confidence, 4),
        "components": {
            "momentum": round(momentum, 2),
            "relative_strength": round(rel, 2),
            "liquidity": round(liquidity, 2),
            "context": round(context, 2),
        },
    }
