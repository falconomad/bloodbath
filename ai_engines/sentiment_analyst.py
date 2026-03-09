import json

POSITIVE_KEYWORDS = {
    "beat",
    "upgrade",
    "raises",
    "surge",
    "contract",
    "approval",
    "growth",
    "partnership",
    "strong",
    "record",
    "ipo",
    "profit",
}
NEGATIVE_KEYWORDS = {
    "downgrade",
    "miss",
    "lawsuit",
    "probe",
    "delay",
    "warning",
    "cuts",
    "loss",
    "fraud",
    "bankruptcy",
    "dilution",
    "offering",
}

def evaluate_sentiment(symbol, news, upcoming_events, macro_calendar):
    """
    Grades a stock strictly on fundamental catalysts and news sentiment.
    Returns a score from 0-100 and a 1-sentence rationale.
    """
    items = news or []
    if not items:
        return {"score": 50, "rationale": "No recent headlines; defaulting to neutral sentiment."}

    pos_hits = 0
    neg_hits = 0
    for article in items:
        txt = f"{article.get('headline','')} {article.get('summary','')}".lower()
        pos_hits += sum(1 for kw in POSITIVE_KEYWORDS if kw in txt)
        neg_hits += sum(1 for kw in NEGATIVE_KEYWORDS if kw in txt)

    net = pos_hits - neg_hits
    score = 50 + (net * 8)
    score = max(0, min(100, int(round(score))))

    rationale = (
        f"Headline keyword balance positive={pos_hits}, negative={neg_hits}. "
        f"Upcoming events context: {upcoming_events}."
    )
    return {"score": score, "rationale": rationale}
