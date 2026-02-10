from __future__ import annotations

from typing import Any

import pandas as pd

from src.analysis.events import score_events
from src.analysis.sentiment import analyze_news_sentiment_details
from src.analysis.technicals import calculate_technicals
from src.api.data_fetcher import get_alpaca_snapshot_features, get_earnings_calendar
from src.pipeline.decision_engine import (
    Signal,
    aggregate_confidence,
    clamp,
    decide,
    normalize_signals,
    weighted_score,
    write_trace,
)
from src.validation.data_validation import (
    validate_earnings_payload,
    validate_micro_features,
    validate_news_headlines,
    validate_price_history,
)


def trend_to_score(trend: str) -> float:
    if trend == "BULLISH":
        return 1.0
    if trend == "BEARISH":
        return -1.0
    return 0.0


def safe_news(news):
    if not news:
        return []
    cleaned = []
    for n in news:
        if isinstance(n, str) and n.strip():
            cleaned.append(n)
        elif isinstance(n, dict) and str(n.get("headline", "")).strip():
            cleaned.append(n)
    return cleaned


def news_quality_factor(headlines):
    n = len(headlines or [])
    if n >= 8:
        return 1.0
    if n >= 4:
        return 0.7
    if n >= 2:
        return 0.45
    return 0.2


def technical_quality_factor(data):
    if data is None or data.empty or "Close" not in data:
        return 0.0
    close = data["Close"].dropna()
    if len(close) >= 80:
        return 1.0
    if len(close) >= 50:
        return 0.75
    if len(close) >= 30:
        return 0.45
    return 0.2


def atr_percent(data, period=14):
    if data is None or data.empty:
        return 0.0
    required = {"High", "Low", "Close"}
    if not required.issubset(set(data.columns)):
        return 0.0

    frame = data[["High", "Low", "Close"]].dropna().copy()
    if len(frame) < period + 1:
        return 0.0

    prev_close = frame["Close"].shift(1)
    tr = pd.concat(
        [
            (frame["High"] - frame["Low"]).abs(),
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period).mean().iloc[-1]
    close = float(frame["Close"].iloc[-1])
    if pd.isna(atr) or close <= 0:
        return 0.0
    return float(atr) / close


def _normalized_module_signals(ticker, data, headlines, dip_meta, cfg):
    dip_meta = dip_meta or {}
    quality_cfg = cfg.get("quality", {})
    risk_cfg = cfg.get("risk", {})

    trend = calculate_technicals(data)
    trend_score = trend_to_score(trend)
    tech_quality = technical_quality_factor(data)
    price_validation = validate_price_history(
        data,
        min_points=quality_cfg.get("min_price_points", 40),
        max_missing_ratio=quality_cfg.get("max_missing_ratio", 0.05),
    )
    price_ok = price_validation.valid
    data_gap_ratio = price_validation.missing_ratio
    trend_signal = Signal(
        name="trend",
        value=trend_score,
        confidence=(0.2 + (0.8 * tech_quality)) if price_ok else 0.0,
        quality_ok=price_ok,
        reason="" if price_ok else price_validation.reason,
    )

    sentiment_details = (
        analyze_news_sentiment_details(headlines)
        if headlines
        else {"score": 0.0, "variance": 0.0, "article_count": 0, "mixed_opinions": False}
    )
    sentiment = float(sentiment_details.get("score", 0.0))
    min_sentiment_articles = int(quality_cfg.get("min_sentiment_articles", 4))
    sentiment_ok, sentiment_reason, sentiment_article_count = validate_news_headlines(headlines, min_sentiment_articles)
    sentiment_variance = min(float(sentiment_details.get("variance", 0.0)), 1.0)
    mixed_opinions = bool(sentiment_details.get("mixed_opinions", False))
    sentiment_signal = Signal(
        name="sentiment",
        value=sentiment,
        confidence=((0.35 + 0.65 * max(abs(sentiment), 0.0)) * (0.75 if mixed_opinions else 1.0)) if sentiment_ok else 0.0,
        quality_ok=sentiment_ok,
        reason="" if sentiment_ok else sentiment_reason,
    )

    earnings = get_earnings_calendar(ticker)
    earnings_ok, earnings_count = validate_earnings_payload(earnings)
    has_upcoming_earnings = earnings_ok and earnings_count > 0
    event_score = float(score_events(headlines, has_upcoming_earnings=has_upcoming_earnings))
    min_news_articles = int(quality_cfg.get("min_news_articles", 3))
    events_ok, events_reason, event_article_count = validate_news_headlines(headlines, min_news_articles)
    events_ok = events_ok or has_upcoming_earnings
    event_signal = Signal(
        name="events",
        value=event_score,
        confidence=0.6 if events_ok else 0.0,
        quality_ok=events_ok,
        reason="" if events_ok else (events_reason or "insufficient_event_context"),
    )

    micro = get_alpaca_snapshot_features(ticker)
    micro_ok, micro_reason, micro_features = validate_micro_features(micro)
    micro_available = micro_ok
    rel_volume = float(micro_features.get("rel_volume", 1.0))
    intraday_return = float(micro_features.get("intraday_return", 0.0))
    micro_quality = float(micro_features.get("quality", 0.0))
    micro_signal_value = clamp((0.6 * intraday_return) + (0.4 * (rel_volume - 1.0)), -1.0, 1.0)
    micro_signal = Signal(
        name="micro",
        value=micro_signal_value,
        confidence=micro_quality if micro_ok else 0.0,
        quality_ok=micro_ok,
        reason="" if micro_ok else micro_reason,
    )

    dip_score = float(dip_meta.get("dip_score", 0.0))
    dip_signal = Signal(
        name="dip",
        value=clamp(dip_score, -1.0, 1.0),
        confidence=0.6 if dip_meta.get("drawdown") is not None else 0.0,
        quality_ok=dip_meta.get("drawdown") is not None,
        reason="" if dip_meta.get("drawdown") is not None else "no_drawdown_context",
    )
    vol_penalty = float(dip_meta.get("volatility_penalty", 0.0))
    vol_scale = max(float(risk_cfg.get("max_atr_pct_for_full_risk", 0.08)), 1e-6)
    volatility_signal = Signal(
        name="volatility",
        value=clamp(vol_penalty / vol_scale, -1.0, 0.0),
        confidence=0.7 if dip_meta.get("drawdown") is not None else 0.0,
        quality_ok=dip_meta.get("drawdown") is not None,
        reason="" if dip_meta.get("drawdown") is not None else "no_volatility_context",
    )

    signals = normalize_signals(
        {
            "trend": trend_signal,
            "sentiment": sentiment_signal,
            "events": event_signal,
            "micro": micro_signal,
            "dip": dip_signal,
            "volatility": volatility_signal,
        }
    )
    return trend, has_upcoming_earnings, signals, {
        "rel_volume": rel_volume,
        "micro_available": micro_available,
        "data_quality_ok": price_ok,
        "data_gap_ratio": data_gap_ratio,
        "atr_pct": atr_percent(data, period=14),
        "article_count": len(headlines),
        "sentiment_article_count": sentiment_article_count,
        "event_article_count": event_article_count,
        "sentiment_variance": sentiment_variance,
        "sentiment_mixed": mixed_opinions,
    }


def generate_recommendation_core(
    ticker: str,
    data,
    headlines,
    dip_meta,
    cycle_idx: int,
    apply_stability_gate: bool,
    cfg: dict[str, Any],
    decision_state: dict[str, dict[str, Any]],
):
    headlines = safe_news(headlines)
    price = float(data["Close"].iloc[-1]) if data is not None and not data.empty and "Close" in data else 0.0
    trend, has_upcoming_earnings, signals, risk_context = _normalized_module_signals(
        ticker=ticker, data=data, headlines=headlines, dip_meta=dip_meta, cfg=cfg
    )
    weights = cfg.get("weights", {})
    score = weighted_score(signals, weights)
    signal_confidence, conflicts = aggregate_confidence(
        signals,
        cfg.get("risk", {}).get("conflict_penalty", 0.2),
        weights=weights,
        max_conflict_drop=cfg.get("risk", {}).get("max_confidence_drop_on_conflict", 0.75),
    )
    signal_confidence = clamp(signal_confidence, 0.0, 1.0)
    event_score = float(signals["events"].value)
    sentiment = float(signals["sentiment"].value)
    event_flag = abs(event_score) > 0.0
    decision, decision_reasons, suggested_position_size = decide(
        ticker=ticker,
        score=score,
        confidence=signal_confidence,
        signals=signals,
        risk_context=risk_context,
        state=decision_state if apply_stability_gate else {},
        cycle_idx=cycle_idx,
        cfg=cfg,
    )
    confidence = signal_confidence * 100.0

    write_trace(
        {
            "ticker": ticker,
            "price": round(price, 6),
            "score": round(score, 6),
            "confidence": round(signal_confidence, 6),
            "decision": decision,
            "decision_reasons": decision_reasons,
            "weights": weights,
            "signals": {k: s.as_dict() for k, s in signals.items()},
            "conflict_ratio": round(conflicts, 6),
            "risk_context": risk_context,
        }
    )

    return {
        "ticker": ticker,
        "trend": trend,
        "sentiment": sentiment,
        "signal_confidence": round(signal_confidence, 4),
        "news_quality": round(news_quality_factor(headlines), 4),
        "technical_quality": round(technical_quality_factor(data), 4),
        "micro_quality": round(signals["micro"].confidence, 4),
        "upcoming_earnings": has_upcoming_earnings,
        "event_detected": event_flag,
        "event_score": event_score,
        "composite_score": round(score, 4),
        "decision": decision,
        "confidence": round(confidence, 2),
        "decision_reasons": decision_reasons,
        "position_size": suggested_position_size,
        "signals": {k: s.as_dict() for k, s in signals.items()},
        "data": data,
    }
