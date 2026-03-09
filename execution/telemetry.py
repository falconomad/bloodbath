from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")


def log_event(event_type: str, payload: dict[str, Any], path: str = "logs/engine_events.jsonl") -> None:
    _append_jsonl(
        path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        },
    )
