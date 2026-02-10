from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Iterable


EVENT_WEIGHTS = {
    "earnings beat": 0.6,
    "beats earnings": 0.6,
    "record profit": 0.6,
    "profit jump": 0.5,
    "acquisition": 0.4,
    "investment": 0.3,
    "launch": 0.25,
    "partnership": 0.25,
    "buyback": 0.35,
    "guidance raise": 0.45,
    "lawsuit": -0.8,
    "investigation": -0.7,
    "downgrade": -0.55,
    "guidance cut": -0.6,
    "misses earnings": -0.6,
    "earnings miss": -0.6,
    "recall": -0.5,
    "bankruptcy": -1.0,
}

TOPIC_LABELS = {
    "earnings": "earnings reports, guidance, or analyst estimates",
    "lawsuits": "lawsuits, investigations, fines, or legal risk",
    "mergers": "mergers, acquisitions, partnerships, or strategic deals",
    "macro events": "macro economy, rates, inflation, geopolitics, or regulation",
}

TOPIC_WEIGHTS = {
    "earnings": 0.25,
    "lawsuits": -0.35,
    "mergers": 0.15,
    "macro events": -0.10,
}

_transformers_spec = importlib.util.find_spec("transformers")
pipeline = None
if _transformers_spec is not None:
    pipeline = importlib.import_module("transformers").pipeline


def _build_zero_shot_model():
    if pipeline is None:
        return None
    # Keep behavior predictable in hosted environments by requiring HF token.
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return None
    try:
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            token=hf_token,
        )
    except Exception:
        return None


zero_shot_model = _build_zero_shot_model()


@lru_cache(maxsize=1024)
def _classify_headline_topic(headline: str) -> tuple[str, float]:
    text = str(headline or "").strip()
    if not text or zero_shot_model is None:
        return "", 0.0
    try:
        labels = list(TOPIC_LABELS.keys())
        result = zero_shot_model(text, labels, hypothesis_template="This headline is about {}.")
        ranked = list(zip(result.get("labels", []) or [], result.get("scores", []) or []))
        if not ranked:
            return "", 0.0
        best_label, best_score = ranked[0]
        return str(best_label), float(best_score)
    except Exception:
        return "", 0.0


def _topic_bonus(news: Iterable[str] | None) -> float:
    if not news:
        return 0.0
    total = 0.0
    for item in news:
        headline = item.get("headline", "") if isinstance(item, dict) else item
        if not isinstance(headline, str):
            continue
        label, confidence = _classify_headline_topic(headline)
        if not label or confidence <= 0:
            continue
        base_weight = float(TOPIC_WEIGHTS.get(label, 0.0))
        # Gate weak classifications to reduce noise.
        strength = max((confidence - 0.45) / 0.55, 0.0)
        total += base_weight * strength
    return total


def _topic_bonus_details(news: Iterable[str] | None) -> tuple[float, dict[str, float], list[dict[str, float | str]]]:
    if not news:
        return 0.0, {}, []
    total = 0.0
    by_topic: dict[str, float] = {}
    per_headline: list[dict[str, float | str]] = []
    for item in news:
        headline = item.get("headline", "") if isinstance(item, dict) else item
        if not isinstance(headline, str):
            continue
        label, confidence = _classify_headline_topic(headline)
        if not label or confidence <= 0:
            continue
        base_weight = float(TOPIC_WEIGHTS.get(label, 0.0))
        strength = max((confidence - 0.45) / 0.55, 0.0)
        contrib = base_weight * strength
        total += contrib
        by_topic[label] = by_topic.get(label, 0.0) + contrib
        per_headline.append(
            {
                "headline": headline[:180],
                "topic": label,
                "confidence": round(float(confidence), 6),
                "contribution": round(float(contrib), 6),
            }
        )
    return total, by_topic, per_headline


def score_events_details(news: Iterable[str] | None, has_upcoming_earnings: bool = False) -> dict[str, object]:
    """Return detailed event scoring components for trace/debug usage."""
    earnings_bonus = 0.12 if has_upcoming_earnings else 0.0
    lexical_total = 0.0
    phrase_hits: dict[str, float] = {}

    if news:
        for headline in news:
            if isinstance(headline, dict):
                headline = headline.get("headline", "")
            if not isinstance(headline, str):
                continue
            text = headline.lower()
            for phrase, weight in EVENT_WEIGHTS.items():
                if phrase in text:
                    lexical_total += weight
                    phrase_hits[phrase] = phrase_hits.get(phrase, 0.0) + float(weight)

    topic_total, topic_by_label, topic_headlines = _topic_bonus_details(news)
    raw_total = earnings_bonus + lexical_total + topic_total
    final = round(max(min(raw_total, 1.0), -1.0), 2)
    return {
        "score": final,
        "earnings_bonus": round(earnings_bonus, 6),
        "lexical_total": round(lexical_total, 6),
        "topic_total": round(topic_total, 6),
        "phrase_hits": {k: round(v, 6) for k, v in phrase_hits.items()},
        "topic_contributions": {k: round(v, 6) for k, v in topic_by_label.items()},
        "topic_headlines": topic_headlines[:12],
    }


def score_events(news: Iterable[str] | None, has_upcoming_earnings: bool = False) -> float:
    """Return weighted event score in [-1, 1] from headline text and earnings context."""
    return float(score_events_details(news, has_upcoming_earnings=has_upcoming_earnings)["score"])


def detect_events(news: Iterable[str] | None, has_upcoming_earnings: bool = False) -> bool:
    """Backwards-compatible boolean event detector."""
    return score_events(news, has_upcoming_earnings=has_upcoming_earnings) != 0.0
