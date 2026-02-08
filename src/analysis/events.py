POSITIVE_EVENT_WEIGHTS = {
    "earnings beat": 0.8,
    "profit": 0.5,
    "acquisition": 0.4,
    "investment": 0.3,
    "launch": 0.2,
    "upgrade": 0.6,
    "guidance raised": 0.7,
}

NEGATIVE_EVENT_WEIGHTS = {
    "earnings miss": -0.8,
    "loss": -0.5,
    "downgrade": -0.6,
    "lawsuit": -0.7,
    "guidance cut": -0.7,
    "investigation": -0.6,
    "recall": -0.5,
}


def calculate_event_score(news):
    if not news:
        return 0.0

    total = 0.0

    for headline in news:
        text = str(headline).lower()

        for phrase, weight in POSITIVE_EVENT_WEIGHTS.items():
            if phrase in text:
                total += weight

        for phrase, weight in NEGATIVE_EVENT_WEIGHTS.items():
            if phrase in text:
                total += weight

    normalized = total / max(len(news), 1)
    return round(max(min(normalized, 1.0), -1.0), 2)


def detect_events(news):
    return calculate_event_score(news) != 0.0
