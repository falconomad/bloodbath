from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class GoalSnapshot:
    start_capital: float
    target_capital: float
    current_capital: float
    started_at: datetime
    deadline_at: datetime
    remaining_capital: float
    days_elapsed: float
    days_remaining: float
    progress_ratio: float
    required_daily_return: float
    pace_multiplier: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "start_capital": round(self.start_capital, 2),
            "target_capital": round(self.target_capital, 2),
            "current_capital": round(self.current_capital, 2),
            "remaining_capital": round(self.remaining_capital, 2),
            "days_elapsed": round(self.days_elapsed, 2),
            "days_remaining": round(self.days_remaining, 2),
            "progress_ratio": round(self.progress_ratio, 6),
            "required_daily_return": round(self.required_daily_return, 6),
            "pace_multiplier": round(self.pace_multiplier, 4),
            "started_at": self.started_at.isoformat(),
            "deadline_at": self.deadline_at.isoformat(),
        }


class GoalPolicy:
    """Converts account progress versus deadline into risk/conviction pacing."""

    def __init__(self, start_capital: float, target_capital: float, horizon_days: int, start_date: str | None = None):
        self.start_capital = max(float(start_capital), 1.0)
        self.target_capital = max(float(target_capital), self.start_capital)
        self.horizon_days = max(int(horizon_days), 1)
        if start_date:
            parsed = datetime.fromisoformat(str(start_date).strip())
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            self.started_at = parsed.astimezone(timezone.utc)
        else:
            self.started_at = datetime.now(timezone.utc)

    def snapshot(self, current_capital: float, now: datetime | None = None) -> GoalSnapshot:
        current = max(float(current_capital), 0.0)
        ts = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)

        deadline = self.started_at + timedelta(days=self.horizon_days)
        elapsed_days = max((ts - self.started_at).total_seconds() / 86_400.0, 0.0)
        remaining_days = max((deadline - ts).total_seconds() / 86_400.0, 0.0)

        total_gap = max(self.target_capital - self.start_capital, 1e-9)
        progress_ratio = min(max((current - self.start_capital) / total_gap, 0.0), 1.5)
        remaining_capital = max(self.target_capital - current, 0.0)

        denom = max(current, 1e-9)
        required_daily_return = 0.0
        if remaining_days > 0 and remaining_capital > 0:
            required_daily_return = (remaining_capital / denom) / remaining_days

        # pace_multiplier > 1 means behind schedule and can tolerate more risk.
        expected_progress = min(elapsed_days / float(self.horizon_days), 1.0)
        behind = max(expected_progress - min(progress_ratio, 1.0), 0.0)
        ahead = max(min(progress_ratio, 1.0) - expected_progress, 0.0)
        pace_multiplier = 1.0 + (behind * 0.50) - (ahead * 0.30)
        pace_multiplier = min(max(pace_multiplier, 0.80), 1.25)

        return GoalSnapshot(
            start_capital=self.start_capital,
            target_capital=self.target_capital,
            current_capital=current,
            started_at=self.started_at,
            deadline_at=deadline,
            remaining_capital=remaining_capital,
            days_elapsed=elapsed_days,
            days_remaining=remaining_days,
            progress_ratio=progress_ratio,
            required_daily_return=required_daily_return,
            pace_multiplier=pace_multiplier,
        )
