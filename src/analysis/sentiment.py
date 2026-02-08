import os
from transformers import pipeline


def _build_sentiment_model():
    """Create the FinBERT pipeline when credentials are available."""
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


def _lexical_fallback_score(text):
    positive_words = {
        "beat", "surge", "growth", "upgrade", "strong", "record", "profit", "bullish"
    }
    negative_words = {
        "miss", "drop", "decline", "downgrade", "weak", "loss", "lawsuit", "bearish"
    }

    tokens = [t.strip(".,:;!?()[]{}\"'").lower() for t in text.split()]
    pos = sum(1 for token in tokens if token in positive_words)
    neg = sum(1 for token in tokens if token in negative_words)

    if pos == 0 and neg == 0:
        return 0.0

    return (pos - neg) / max(pos + neg, 1)



def analyze_news_sentiment(news_list):
    if not news_list:
        return 0

    if sentiment_model is None:
        fallback = sum(_lexical_fallback_score(item) for item in news_list) / len(news_list)
        return round(max(min(fallback, 1.0), -1.0), 2)

    try:
        results = sentiment_model(news_list)
    except Exception:
        fallback = sum(_lexical_fallback_score(item) for item in news_list) / len(news_list)
        return round(max(min(fallback, 1.0), -1.0), 2)

    score = 0
    for r in results:
        label = str(r.get("label", "")).lower()
        value = float(r.get("score", 0.0))
        if label == "positive":
            score += value
        elif label == "negative":
            score -= value

    return round(max(min(score / len(results), 1.0), -1.0), 2)
