import json
from datetime import datetime, timezone

POSITIVE_KEYWORDS = {
    "beat": 2.0,
    "upgrade": 2.0,
    "raises": 1.5,
    "surge": 1.5,
    "contract": 2.0,
    "approval": 2.0,
    "growth": 1.0,
    "partnership": 1.3,
    "strong": 1.0,
    "record": 1.2,
    "ipo": 1.0,
    "profit": 1.2,
    "guidance": 1.3,
}
NEGATIVE_KEYWORDS = {
    "downgrade": 2.0,
    "miss": 2.0,
    "lawsuit": 2.0,
    "probe": 2.0,
    "delay": 1.2,
    "warning": 1.8,
    "cuts": 1.5,
    "loss": 1.3,
    "fraud": 2.2,
    "bankruptcy": 2.5,
    "dilution": 2.0,
    "offering": 1.8,
}


def _parse_date(s: str):
    try:
        val = str(s or "").replace("Z", "+00:00")
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _recency_weight(iso_dt: str):
    dt = _parse_date(iso_dt)
    hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    if hours <= 6:
        return 1.0
    if hours <= 24:
        return 0.7
    if hours <= 72:
        return 0.4
    return 0.2


def _source_weight(source: str):
    s = str(source or "").lower()
    if "sec" in s:
        return 1.2
    if "reuters" in s or "bloomberg" in s or "wsj" in s:
        return 1.1
    if "yahoo" in s or "alpaca" in s:
        return 1.0
    return 0.9

def evaluate_sentiment(symbol, news, upcoming_events, macro_calendar):
    """
    Grades a stock strictly on fundamental catalysts and news sentiment.
    Returns a score from 0-100 and a 1-sentence rationale.
    """
    items = news or []
    if not items:
        return {"score": 50, "confidence": 0.35, "rationale": "No recent headlines; defaulting to neutral sentiment."}

    pos_hits = 0.0
    neg_hits = 0.0
    signed_scores = []
    for article in items:
        txt = f"{article.get('headline','')} {article.get('summary','')}".lower()
        recency_w = _recency_weight(article.get("date", ""))
        source_w = _source_weight(article.get("source", ""))
        weight = recency_w * source_w
        p = sum(w for kw, w in POSITIVE_KEYWORDS.items() if kw in txt)
        n = sum(w for kw, w in NEGATIVE_KEYWORDS.items() if kw in txt)
        pos_hits += p * weight
        neg_hits += n * weight
        signed_scores.append((p - n) * weight)

    event_bonus = 0.0
    if isinstance(upcoming_events, dict) and upcoming_events.get("earnings_upcoming_7d", False):
        event_bonus += 1.0

    macro_penalty = 0.0
    if isinstance(macro_calendar, dict):
        if str(macro_calendar.get("risk_level", "low")).lower() == "high":
            macro_penalty = 1.0
        elif str(macro_calendar.get("risk_level", "low")).lower() == "medium":
            macro_penalty = 0.4

    net = (pos_hits - neg_hits) + event_bonus - macro_penalty
    score = 50 + (net * 8)
    score = max(0, min(100, int(round(score))))

    variance = 0.0
    if signed_scores:
        avg = sum(signed_scores) / len(signed_scores)
        variance = sum(abs(x - avg) for x in signed_scores) / len(signed_scores)
    disagreement_penalty = min(0.35, variance * 0.08)
    confidence = 0.40 + min(0.45, (len(items) / 6.0) * 0.45) + min(0.20, abs(net) * 0.03) - disagreement_penalty
    confidence = max(0.15, min(0.95, confidence))

    rationale = (
        f"Weighted headline sentiment positive={pos_hits:.2f}, negative={neg_hits:.2f}, "
        f"net={net:.2f}, macro_risk={macro_calendar.get('risk_level','unknown') if isinstance(macro_calendar, dict) else 'unknown'}."
    )
    return {"score": score, "confidence": round(confidence, 4), "rationale": rationale}
