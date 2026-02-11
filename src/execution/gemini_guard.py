from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone, timedelta
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
        self._base_daily_cap = self._default_daily_cap(self.model)
        self.max_tickers_per_request = self._default_batch_size(self.model)
        self.max_prompt_chars = 7000

    def enabled(self) -> bool:
        return bool(self.api_key) and self.max_calls_per_cycle > 0 and self.max_calls_per_day > 0

    def status_snapshot(self) -> dict[str, Any]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date = str(self._state.get("date", ""))
        calls_today = int(self._state.get("calls_today", 0) or 0) if date == today else 0
        tokens_today = int(self._state.get("tokens_today", 0) or 0) if date == today else 0
        dynamic_cap = int(self._state.get("dynamic_daily_cap", self._base_daily_cap) or self._base_daily_cap)
        hard_cap = min(self.max_calls_per_day, dynamic_cap)
        cache = self._state.get("verdict_cache", {}) or {}
        return {
            "enabled": self.enabled(),
            "model": self.model,
            "calls_today": calls_today,
            "tokens_today": tokens_today,
            "hard_cap": hard_cap,
            "remaining_calls_today": max(hard_cap - calls_today, 0),
            "cooldown_until": str(self._state.get("cooldown_until", "") or ""),
            "last_call_ts": str(self._state.get("last_call_ts", "") or ""),
            "last_rate_limit_ts": str(self._state.get("last_rate_limit_ts", "") or ""),
            "cache_size": len(cache),
            "max_tickers_per_request": self.max_tickers_per_request,
        }

    def _default_daily_cap(self, model: str) -> int:
        m = str(model or "").lower()
        if "flash-lite" in m:
            return 100
        if "flash" in m:
            return 60
        return 30

    def _default_batch_size(self, model: str) -> int:
        m = str(model or "").lower()
        if "flash-lite" in m:
            return 8
        if "flash" in m:
            return 6
        return 4

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
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if str(self._state.get("date", "")) != today:
            self._state["date"] = today
            self._state["calls_today"] = 0
            self._state["tokens_today"] = 0
            self._state["dynamic_daily_cap"] = self._base_daily_cap
            self._state.pop("cooldown_until", None)
            self._state["verdict_cache"] = {}
        cooldown_until = str(self._state.get("cooldown_until", "") or "").strip()
        if cooldown_until:
            try:
                if now < datetime.fromisoformat(cooldown_until):
                    return False
            except Exception:
                pass
        dynamic_cap = int(self._state.get("dynamic_daily_cap", self._base_daily_cap) or self._base_daily_cap)
        hard_cap = min(self.max_calls_per_day, dynamic_cap)
        allowed = int(self._state.get("calls_today", 0)) < hard_cap
        if not allowed:
            print(f"[call][gemini] blocked budget_exhausted calls_today={self._state.get('calls_today', 0)} cap={hard_cap}")
        return allowed

    def _record_call(self) -> None:
        self._state["calls_today"] = int(self._state.get("calls_today", 0)) + 1
        self._state["last_call_ts"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def _record_tokens(self, tokens: int) -> None:
        if tokens <= 0:
            return
        self._state["tokens_today"] = int(self._state.get("tokens_today", 0)) + int(tokens)
        self._save_state()

    def _record_rate_limit(self) -> None:
        now = datetime.now(timezone.utc)
        current = int(self._state.get("dynamic_daily_cap", self._base_daily_cap) or self._base_daily_cap)
        reduced = max(int(current * 0.8), 5)
        self._state["dynamic_daily_cap"] = reduced
        self._state["cooldown_until"] = (now.replace(microsecond=0) + timedelta(minutes=20)).isoformat()
        self._state["last_rate_limit_ts"] = now.isoformat()
        self._save_state()
        print(
            "[result][gemini] rate_limited "
            f"dynamic_daily_cap={self._state.get('dynamic_daily_cap')} cooldown_until={self._state.get('cooldown_until')}"
        )

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

    def _prompt_for_many(self, rows: list[dict[str, Any]]) -> str:
        lines = [
            "You are a strict risk gate for an automated trading engine.",
            "Return JSON only with this schema:",
            '{"verdicts":[{"ticker":"TICKER","action":"ALLOW|BLOCK|REDUCE","confidence":0.0,"reason":"...","size_mult":1.0}]}',
            "Prefer BLOCK if risk is unclear.",
            "",
            "Candidates:",
        ]
        for row in rows:
            decision = str(row.get("decision", "HOLD")).upper()
            reasons = ", ".join([str(x) for x in (row.get("decision_reasons", []) or [])][:10])
            lines.append(
                (
                    f"- ticker={row.get('ticker','')} decision={decision} score={float(row.get('score',0.0)):.4f} "
                    f"signal_conf={float(row.get('signal_confidence',0.0)):.4f} atr_pct={float(row.get('atr_pct',0.0)):.5f} "
                    f"growth_20d={float(row.get('growth_20d',0.0)):.4f} daily_return={float(row.get('daily_return',0.0)):.6f} "
                    f"sentiment={float(row.get('sentiment',0.0)):.4f} position_size={float(row.get('position_size',0.0)):.4f} "
                    f"mover_bucket={row.get('mover_bucket','')} social_posts={int(row.get('social_post_count',0) or 0)} "
                    f"market_news={int(row.get('market_news_count',0) or 0)} "
                    f"reasons={reasons}"
                )
            )
        return "\n".join(lines)

    def _usage_tokens(self, payload: dict[str, Any]) -> int:
        md = payload.get("usageMetadata", {}) or {}
        vals = [
            md.get("totalTokenCount", 0),
            md.get("promptTokenCount", 0),
            md.get("candidatesTokenCount", 0),
        ]
        for v in vals:
            try:
                n = int(v)
                if n > 0:
                    return n
            except Exception:
                continue
        return 0

    def _fingerprint(self, row: dict[str, Any]) -> str:
        compact = {
            "t": str(row.get("ticker", "")).upper(),
            "s": round(float(row.get("score", 0.0) or 0.0), 4),
            "c": round(float(row.get("signal_confidence", 0.0) or 0.0), 4),
            "a": round(float(row.get("atr_pct", 0.0) or 0.0), 5),
            "g": round(float(row.get("growth_20d", 0.0) or 0.0), 4),
            "d": round(float(row.get("daily_return", 0.0) or 0.0), 5),
            "p": round(float(row.get("position_size", 0.0) or 0.0), 4),
        }
        raw = json.dumps(compact, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def _cache_get(self, row: dict[str, Any]) -> dict[str, Any] | None:
        cache = self._state.get("verdict_cache", {}) or {}
        key = self._fingerprint(row)
        hit = cache.get(key)
        if not isinstance(hit, dict):
            return None
        exp = str(hit.get("expires_at", "")).strip()
        if not exp:
            return None
        try:
            if datetime.now(timezone.utc) >= datetime.fromisoformat(exp):
                return None
        except Exception:
            return None
        return dict(hit.get("verdict", {}) or {})

    def _cache_put(self, row: dict[str, Any], verdict: dict[str, Any], ttl_minutes: int = 90) -> None:
        key = self._fingerprint(row)
        cache = self._state.get("verdict_cache", {}) or {}
        cache[key] = {
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=max(ttl_minutes, 5))).isoformat(),
            "verdict": verdict,
        }
        if len(cache) > 400:
            keys = list(cache.keys())
            for k in keys[: max(len(cache) - 400, 0)]:
                cache.pop(k, None)
        self._state["verdict_cache"] = cache
        self._save_state()

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

    def _parse_json_result(self, text: str) -> dict[str, Any] | list[dict[str, Any]] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) or isinstance(obj, list):
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
        resp = requests.post(url, params={"key": self.api_key}, json=body, timeout=self.timeout_seconds)
        if resp.status_code == 429:
            self._record_rate_limit()
            return None
        resp.raise_for_status()
        self._record_call()
        text = self._extract_text(resp.json() or {})
        return self._parse_json_result(text)

    def _gemini_check_many(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if not rows:
            return []
        if not self.enabled() or not self._can_call():
            return None
        tickers = ",".join([str(r.get("ticker", "")).upper() for r in rows if str(r.get("ticker", "")).strip()])
        print(f"[call][gemini] model={self.model} batch={len(rows)} tickers={tickers}")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": self._prompt_for_many(rows)}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        resp = requests.post(url, params={"key": self.api_key}, json=body, timeout=self.timeout_seconds)
        if resp.status_code == 429:
            self._record_rate_limit()
            return None
        resp.raise_for_status()
        self._record_call()
        payload = resp.json() or {}
        self._record_tokens(self._usage_tokens(payload))
        print(
            "[result][gemini] success "
            f"status={resp.status_code} calls_today={self._state.get('calls_today', 0)} "
            f"tokens_today={self._state.get('tokens_today', 0)}"
        )
        text = self._extract_text(payload)
        parsed = self._parse_json_result(text)
        if isinstance(parsed, dict):
            verdicts = parsed.get("verdicts", [])
            if isinstance(verdicts, list):
                return [x for x in verdicts if isinstance(x, dict)]
            return []
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        return []

    def apply(self, analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled():
            return analyses
        out = [dict(x) for x in analyses]
        buys = [x for x in out if str(x.get("decision", "HOLD")).upper() == "BUY"]
        buys.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        pending: list[dict[str, Any]] = []
        cache_hits = 0
        for row in buys:
            cached = self._cache_get(row)
            if cached:
                cache_hits += 1
                row["gemini_cached"] = True
                row["gemini_action"] = str(cached.get("action", "ALLOW")).upper()
                row["gemini_confidence"] = round(float(cached.get("confidence", 0.0) or 0.0), 4)
                action = row["gemini_action"]
                reasons = list(row.get("decision_reasons", []) or [])
                if action == "BLOCK":
                    row["decision"] = "HOLD"
                    reasons.append("gemini:block:cache")
                elif action == "REDUCE":
                    m = max(0.0, min(float(cached.get("size_mult", 1.0) or 1.0), 1.0))
                    row["position_size"] = float(row.get("position_size", 0.0)) * m
                    reasons.append(f"gemini:reduce:{m:.2f}:cache")
                else:
                    reasons.append("gemini:allow:cache")
                row["decision_reasons"] = reasons
            else:
                pending.append(row)
        if buys:
            print(
                "[cycle][gemini][plan] "
                f"buys={len(buys)} pending={len(pending)} cache_hits={cache_hits} "
                f"max_tickers_per_request={self.max_tickers_per_request}"
            )
        # Free-tier optimization: keep request count tiny; pack multiple tickers per call.
        request_budget = min(self.max_calls_per_cycle, 1)
        idx = 0
        for _ in range(request_budget):
            if idx >= len(pending):
                break
            batch = []
            while idx < len(pending) and len(batch) < max(self.max_tickers_per_request, 1):
                candidate = pending[idx]
                trial = batch + [candidate]
                if len(self._prompt_for_many(trial)) > self.max_prompt_chars and batch:
                    break
                batch.append(candidate)
                idx += 1
            if not batch:
                break
            try:
                verdicts = self._gemini_check_many(batch)
            except Exception as exc:
                for row in batch:
                    reasons = list(row.get("decision_reasons", []) or [])
                    reasons.append(f"gemini:error:{type(exc).__name__}")
                    row["decision_reasons"] = reasons
                continue
            if not verdicts:
                continue
            by_ticker = {str(v.get("ticker", "")).upper(): v for v in verdicts if str(v.get("ticker", "")).strip()}
            for row in batch:
                verdict = by_ticker.get(str(row.get("ticker", "")).upper())
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
                self._cache_put(
                    row,
                    {
                        "action": action,
                        "confidence": conf,
                        "size_mult": size_mult,
                        "reason": reason,
                    },
                    ttl_minutes=90,
                )
        return out
