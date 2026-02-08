
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def analyze_news_sentiment(news_list):

    if not news_list:
        return 0

    results = sentiment_model(news_list)

    score = 0
    for r in results:
        if r["label"] == "positive":
            score += r["score"]
        elif r["label"] == "negative":
            score -= r["score"]

    return round(score / len(results), 2)
