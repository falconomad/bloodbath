import os
import importlib
from datetime import datetime, timezone


_transformers_spec = importlib.util.find_spec("transformers")
pipeline = None
if _transformers_spec is not None:
    pipeline = importlib.import_module("transformers").pipeline


SOURCE_RELIABILITY = {
    "reuters": 1.0,
    "bloomberg": 0.98,
    "wsj": 0.95,
    "marketwatch": 0.88,
    "benzinga": 0.8,
    "seekingalpha": 0.78,
    "yahoo": 0.75,
}
DEFAULT_SOURCE_RELIABILITY = 0.7
RECENCY_HALF_LIFE_DAYS = 3.0
OUTLIER_CLIP_STD = 2.0


def _build_sentiment_model():
    """Create the FinBERT pipeline when credentials are available."""
    if pipeline is None:
        return None

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return None

    try:
        return pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            token=hf_token,
        )
    except Exception:
        return None


sentiment_model = _build_sentiment_model()


def _parse_published_at(item):
    if not isinstance(item, dict):
        return None
    raw = item.get("datetime") or item.get("published_at") or item.get("time")
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        text = str(raw).strip()
        if not text:
            return None
        text = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_news_items(news_list, limit=12):
    normalized = []
    seen = set()

    for item in news_list:
        if isinstance(item, str):
            headline = " ".join(item.strip().split())
            source = ""
            published_at = None
        elif isinstance(item, dict):
            headline = " ".join(str(item.get("headline", "")).strip().split())
            source = str(item.get("source", "")).strip().lower()
            published_at = _parse_published_at(item)
        else:
            continue

        if not headline:
            continue
        key = headline.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"headline": headline, "source": source, "published_at": published_at})
        if len(normalized) >= limit:
            break

    return normalized


def _prepare_headlines(news_list, limit=12):
    return [item["headline"] for item in _normalize_news_items(news_list, limit=limit)]


def _lexical_fallback_score(text):
    positive_weights = {
        "beat": 1.0,
        "surge": 1.0,
        "growth": 0.8,
        "upgrade": 1.0,
        "strong": 0.7,
        "record": 0.8,
        "profit": 0.7,
        "bullish": 1.0,
        "outperform": 1.0,
        "guidance raise": 1.1,
        "buyback": 0.9,
    }
    negative_weights = {
        "miss": 1.0,
        "drop": 0.9,
        "decline": 0.8,
        "downgrade": 1.0,
        "weak": 0.7,
        "loss": 0.9,
        "lawsuit": 1.1,
        "bearish": 1.0,
        "investigation": 1.1,
        "guidance cut": 1.2,
        "recall": 1.0,
    }

    clean = str(text).lower()
    tokens = [t.strip(".,:;!?()[]{}\"'").lower() for t in clean.split()]
    pos = sum(positive_weights.get(token, 0.0) for token in tokens)
    neg = sum(negative_weights.get(token, 0.0) for token in tokens)

    for phrase, weight in positive_weights.items():
        if " " in phrase and phrase in clean:
            pos += weight
    for phrase, weight in negative_weights.items():
        if " " in phrase and phrase in clean:
            neg += weight

    if pos == 0 and neg == 0:
        return 0.0

    return (pos - neg) / max(pos + neg, 1)


def _prediction_to_score(prediction):
    label = str(prediction.get("label", "")).lower()
    value = float(prediction.get("score", 0.0))
    contribution = max((value - 0.5) * 2.0, 0.0)
    if label == "positive":
        return contribution
    if label == "negative":
        return -contribution
    return 0.0


def _recency_weight(published_at):
    if published_at is None:
        return 0.85
    age_days = max((datetime.now(timezone.utc) - published_at).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def _source_weight(source):
    if not source:
        return DEFAULT_SOURCE_RELIABILITY
    lower = source.lower()
    for key, value in SOURCE_RELIABILITY.items():
        if key in lower:
            return value
    return DEFAULT_SOURCE_RELIABILITY


def _clip_outliers(scores):
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    if std <= 1e-9:
        return scores
    lo = mean - (OUTLIER_CLIP_STD * std)
    hi = mean + (OUTLIER_CLIP_STD * std)
    return [max(min(s, hi), lo) for s in scores]


def analyze_news_sentiment_details(news_list):
    items = _normalize_news_items(news_list)
    if not items:
        return {"score": 0.0, "variance": 0.0, "article_count": 0, "mixed_opinions": False}

    headlines = [it["headline"] for it in items]

    if sentiment_model is None:
        raw_scores = [_lexical_fallback_score(h) for h in headlines]
    else:
        try:
            results = sentiment_model(headlines)
            raw_scores = [_prediction_to_score(r) for r in results]
        except Exception:
            raw_scores = [_lexical_fallback_score(h) for h in headlines]

    clipped = _clip_outliers(raw_scores)
    weights = []
    for item in items:
        weights.append(_recency_weight(item["published_at"]) * _source_weight(item["source"]))

    total_w = sum(weights)
    if total_w <= 0:
        weighted = 0.0
    else:
        weighted = sum(s * w for s, w in zip(clipped, weights)) / total_w

    mean = sum(clipped) / len(clipped)
    variance = sum((s - mean) ** 2 for s in clipped) / len(clipped)
    sign_conflict = any(s > 0.1 for s in clipped) and any(s < -0.1 for s in clipped)
    mixed = sign_conflict and variance > 0.08

    if mixed:
        weighted *= 0.7

    return {
        "score": round(max(min(weighted, 1.0), -1.0), 4),
        "variance": round(max(variance, 0.0), 6),
        "article_count": len(items),
        "mixed_opinions": mixed,
    }


def analyze_news_sentiment(news_list):
    details = analyze_news_sentiment_details(news_list)
    return round(float(details.get("score", 0.0)), 2)
