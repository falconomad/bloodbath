import os
import importlib


_transformers_spec = importlib.util.find_spec("transformers")
pipeline = None
if _transformers_spec is not None:
    pipeline = importlib.import_module("transformers").pipeline


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


def _prepare_headlines(news_list, limit=12):
    prepared = []
    seen = set()

    for item in news_list:
        if not isinstance(item, str):
            continue
        text = " ".join(item.strip().split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        prepared.append(text)
        if len(prepared) >= limit:
            break

    return prepared


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

    # Phrase-level boosts that token splits can miss.
    for phrase, weight in positive_weights.items():
        if " " in phrase and phrase in clean:
            pos += weight
    for phrase, weight in negative_weights.items():
        if " " in phrase and phrase in clean:
            neg += weight

    if pos == 0 and neg == 0:
        return 0.0

    return (pos - neg) / max(pos + neg, 1)



def analyze_news_sentiment(news_list):
    headlines = _prepare_headlines(news_list)
    if not headlines:
        return 0

    if sentiment_model is None:
        fallback = sum(_lexical_fallback_score(item) for item in headlines) / len(headlines)
        return round(max(min(fallback, 1.0), -1.0), 2)

    try:
        results = sentiment_model(headlines)
    except Exception:
        fallback = sum(_lexical_fallback_score(item) for item in headlines) / len(headlines)
        return round(max(min(fallback, 1.0), -1.0), 2)

    score = 0
    for r in results:
        label = str(r.get("label", "")).lower()
        value = float(r.get("score", 0.0))

        # Convert confidence into centered contribution to avoid overstating weak labels.
        contribution = max((value - 0.5) * 2.0, 0.0)
        if label == "positive":
            score += contribution
        elif label == "negative":
            score -= contribution

    return round(max(min(score / len(results), 1.0), -1.0), 2)
