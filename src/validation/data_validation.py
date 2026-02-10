from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class DataQuality:
    valid: bool
    reason: str
    missing_ratio: float
    points: int


def _extract_close_values(data: Any) -> list[float | None]:
    if data is None:
        return []

    # pandas DataFrame-like path.
    try:
        if hasattr(data, "empty") and bool(data.empty):
            return []
        close = data["Close"]
    except Exception:
        close = None

    if close is None and isinstance(data, dict):
        close = data.get("Close")

    if close is None:
        return []

    try:
        values = list(close)
    except Exception:
        return []
    return values


def validate_price_history(data: Any, min_points: int = 40, max_missing_ratio: float = 0.05) -> DataQuality:
    close_values = _extract_close_values(data)
    if not close_values:
        return DataQuality(valid=False, reason="missing_close_series", missing_ratio=1.0, points=0)

    points = int(len(close_values))
    if points < int(min_points):
        missing = sum(1 for v in close_values if v is None)
        return DataQuality(valid=False, reason="insufficient_points", missing_ratio=(missing / max(points, 1)), points=points)

    missing = sum(1 for v in close_values if v is None)
    missing_ratio = missing / max(points, 1)
    if missing_ratio > float(max_missing_ratio):
        return DataQuality(valid=False, reason="too_many_gaps", missing_ratio=missing_ratio, points=points)

    non_na = [v for v in close_values if v is not None]
    if not non_na:
        return DataQuality(valid=False, reason="all_close_missing", missing_ratio=1.0, points=points)

    try:
        if any(float(v) <= 0 for v in non_na):
            return DataQuality(valid=False, reason="non_positive_close", missing_ratio=missing_ratio, points=points)
    except Exception:
        return DataQuality(valid=False, reason="invalid_close_values", missing_ratio=missing_ratio, points=points)

    return DataQuality(valid=True, reason="", missing_ratio=missing_ratio, points=points)


def validate_micro_features(micro: dict[str, Any] | None) -> tuple[bool, str, dict[str, float]]:
    if not isinstance(micro, dict):
        return False, "invalid_micro_payload", {"rel_volume": 1.0, "intraday_return": 0.0, "quality": 0.0}

    available = bool(micro.get("available", False))
    rel_volume = _to_float(micro.get("rel_volume", 1.0), 1.0)
    intraday_return = _to_float(micro.get("intraday_return", 0.0), 0.0)
    quality = _to_float(micro.get("quality", 0.0), 0.0)

    if not available:
        return False, "micro_unavailable", {"rel_volume": rel_volume, "intraday_return": intraday_return, "quality": quality}

    if rel_volume <= 0:
        return False, "invalid_rel_volume", {"rel_volume": 1.0, "intraday_return": intraday_return, "quality": quality}

    if quality <= 0:
        return False, "invalid_micro_quality", {"rel_volume": rel_volume, "intraday_return": intraday_return, "quality": 0.0}

    return True, "", {"rel_volume": rel_volume, "intraday_return": intraday_return, "quality": quality}


def validate_news_headlines(headlines: list[str] | None, min_articles: int) -> tuple[bool, str, int]:
    clean = [h for h in (headlines or []) if isinstance(h, str) and h.strip()]
    count = len(clean)
    if count < int(min_articles):
        return False, "too_few_articles", count
    return True, "", count


def validate_earnings_payload(earnings: Any) -> tuple[bool, int]:
    try:
        count = len(earnings)
    except Exception:
        return False, 0
    return True, int(count)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
