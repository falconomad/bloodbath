from __future__ import annotations

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


def score_events(news: Iterable[str] | None, has_upcoming_earnings: bool = False) -> float:
    """Return weighted event score in [-1, 1] from headline text and earnings context."""
    total = 0.12 if has_upcoming_earnings else 0.0

    if news:
        for headline in news:
            if not isinstance(headline, str):
                continue
            text = headline.lower()
            for phrase, weight in EVENT_WEIGHTS.items():
                if phrase in text:
                    total += weight

    return round(max(min(total, 1.0), -1.0), 2)


def detect_events(news: Iterable[str] | None, has_upcoming_earnings: bool = False) -> bool:
    """Backwards-compatible boolean event detector."""
    return score_events(news, has_upcoming_earnings=has_upcoming_earnings) != 0.0
