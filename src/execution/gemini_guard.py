from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


class GeminiGuard:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_calls_per_cycle: int = 1,
        max_calls_per_day: int = 80,
        timeout_seconds: float = 8.0,
        state_path: str = "logs/gemini_guard_state.json",
    ):
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "gemini-2.5-flash-lite").strip()
        self.max_calls_per_cycle = max(int(max_calls_per_cycle), 0)
        self.max_calls_per_day = max(int(max_calls_per_day), 0)
        self.timeout_seconds = float(timeout_seconds)
        self.state_path = Path(state_path)
        self._state = self._load_state()

    def enabled(self) -> bool:
        return bool(self.api_key) and self.max_calls_per_cycle > 0 and self.max_calls_per_day > 0

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"date": "", "calls_today": 0}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"date": "", "calls_today": 0}
        except Exception:
            return {"date": "", "calls_today": 0}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")

    def _can_call(self) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if str(self._state.get("date", "")) != today:
            self._state["date"] = today
            self._state["calls_today"] = 0
        return int(self._state.get("calls_today", 0)) < self.max_calls_per_day

    def _record_call(self) -> None:
        self._state["calls_today"] = int(self._state.get("calls_today", 0)) + 1
        self._state["last_call_ts"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def _prompt_for(self, row: dict[str, Any]) -> str:
        decision = str(row.get("decision", "HOLD")).upper()
        reasons = ", ".join([str(x) for x in (row.get("decision_reasons", []) or [])][:8])
        return (
            "You are a strict risk gate for an automated trading engine.\n"
            "Return JSON only with keys: action, confidence, reason, size_mult.\n"
            "Valid actions: ALLOW, BLOCK, REDUCE.\n"
            "size_mult must be in [0,1], use 1 for ALLOW/BLOCK unless REDUCE.\n"
            "Prefer BLOCK if risk is unclear.\n\n"
            f"ticker={row.get('ticker','')}\n"
            f"proposed_decision={decision}\n"
            f"score={float(row.get('score',0.0)):.4f}\n"
            f"signal_confidence={float(row.get('signal_confidence',0.0)):.4f}\n"
            f"atr_pct={float(row.get('atr_pct',0.0)):.5f}\n"
            f"growth_20d={float(row.get('growth_20d',0.0)):.4f}\n"
            f"daily_return={float(row.get('daily_return',0.0)):.6f}\n"
            f"sentiment={float(row.get('sentiment',0.0)):.4f}\n"
            f"position_size={float(row.get('position_size',0.0)):.4f}\n"
            f"decision_reasons={reasons}\n"
        )

    def _extract_text(self, payload: dict[str, Any]) -> str:
        try:
            candidates = payload.get("candidates", []) or []
            if not candidates:
                return ""
            parts = (((candidates[0] or {}).get("content", {}) or {}).get("parts", []) or [])
            for p in parts:
                if "text" in p:
                    return str(p.get("text", ""))
        except Exception:
            return ""
        return ""

    def _parse_json_result(self, text: str) -> dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def _gemini_check(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled() or not self._can_call():
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": self._prompt_for(row)}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        resp = requests.post(
            url,
            params={"key": self.api_key},
            json=body,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        self._record_call()
        text = self._extract_text(resp.json() or {})
        return self._parse_json_result(text)

    def apply(self, analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled():
            return analyses
        out = [dict(x) for x in analyses]
        buys = [x for x in out if str(x.get("decision", "HOLD")).upper() == "BUY"]
        buys.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        budget = min(self.max_calls_per_cycle, len(buys))
        for row in buys[:budget]:
            try:
                verdict = self._gemini_check(row)
            except Exception as exc:
                reasons = list(row.get("decision_reasons", []) or [])
                reasons.append(f"gemini:error:{type(exc).__name__}")
                row["decision_reasons"] = reasons
                continue
            if not verdict:
                continue
            action = str(verdict.get("action", "ALLOW")).strip().upper()
            reason = str(verdict.get("reason", "")).strip()[:120]
            conf = float(verdict.get("confidence", 0.0) or 0.0)
            size_mult = float(verdict.get("size_mult", 1.0) or 1.0)
            size_mult = max(0.0, min(size_mult, 1.0))
            reasons = list(row.get("decision_reasons", []) or [])
            if action == "BLOCK":
                row["decision"] = "HOLD"
                reasons.append(f"gemini:block:{reason}" if reason else "gemini:block")
            elif action == "REDUCE":
                row["position_size"] = float(row.get("position_size", 0.0)) * size_mult
                reasons.append(f"gemini:reduce:{size_mult:.2f}")
            else:
                reasons.append("gemini:allow")
            row["decision_reasons"] = reasons
            row["gemini_action"] = action
            row["gemini_confidence"] = round(conf, 4)
        return out
