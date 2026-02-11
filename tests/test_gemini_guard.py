import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.execution.gemini_guard import GeminiGuard


class _FakeGeminiGuard(GeminiGuard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queued = []

    def queue(self, item):
        self._queued.append(item)

    def _gemini_check(self, row):  # noqa: D401
        if not self._queued:
            return None
        self._record_call()
        return self._queued.pop(0)

    def _gemini_check_many(self, rows):  # noqa: D401
        if not self._queued:
            return None
        self._record_call()
        item = self._queued.pop(0)
        if isinstance(item, list):
            return item
        if isinstance(item, dict):
            return [item]
        return None


class GeminiGuardTests(unittest.TestCase):
    def test_blocks_top_buy_only_with_cycle_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = _FakeGeminiGuard(
                api_key="x",
                max_calls_per_cycle=1,
                max_calls_per_day=5,
                state_path=str(Path(tmp) / "gemini_state.json"),
            )
            guard.queue({"ticker": "AAA", "action": "BLOCK", "confidence": 0.9, "reason": "high risk", "size_mult": 1.0})
            analyses = [
                {"ticker": "AAA", "decision": "BUY", "score": 1.2, "position_size": 0.2, "decision_reasons": []},
                {"ticker": "BBB", "decision": "BUY", "score": 0.8, "position_size": 0.2, "decision_reasons": []},
            ]
            out = guard.apply(analyses)
            row_aaa = [x for x in out if x["ticker"] == "AAA"][0]
            row_bbb = [x for x in out if x["ticker"] == "BBB"][0]
            self.assertEqual(row_aaa["decision"], "HOLD")
            self.assertEqual(row_bbb["decision"], "BUY")

    def test_reduce_scales_position_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = _FakeGeminiGuard(
                api_key="x",
                max_calls_per_cycle=1,
                max_calls_per_day=5,
                state_path=str(Path(tmp) / "gemini_state.json"),
            )
            guard.queue({"ticker": "AAA", "action": "REDUCE", "confidence": 0.7, "reason": "vol", "size_mult": 0.4})
            analyses = [{"ticker": "AAA", "decision": "BUY", "score": 1.1, "position_size": 0.2, "decision_reasons": []}]
            out = guard.apply(analyses)
            self.assertAlmostEqual(float(out[0]["position_size"]), 0.08, places=6)

    def test_cooldown_blocks_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = _FakeGeminiGuard(
                api_key="x",
                max_calls_per_cycle=1,
                max_calls_per_day=50,
                state_path=str(Path(tmp) / "gemini_state.json"),
            )
            guard._state["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            guard._state["cooldown_until"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
            guard._save_state()
            out = guard.apply([{"ticker": "AAA", "decision": "BUY", "score": 2.0, "position_size": 0.2, "decision_reasons": []}])
            self.assertEqual(out[0]["decision"], "BUY")

    def test_batch_can_handle_multiple_tickers_one_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = _FakeGeminiGuard(
                api_key="x",
                max_calls_per_cycle=1,
                max_calls_per_day=50,
                state_path=str(Path(tmp) / "gemini_state.json"),
                model="gemini-2.5-flash-lite",
            )
            guard.queue(
                [
                    {"ticker": "AAA", "action": "BLOCK", "confidence": 0.8, "reason": "news risk", "size_mult": 1.0},
                    {"ticker": "BBB", "action": "REDUCE", "confidence": 0.7, "reason": "vol", "size_mult": 0.5},
                ]
            )
            analyses = [
                {"ticker": "AAA", "decision": "BUY", "score": 2.0, "position_size": 0.2, "decision_reasons": []},
                {"ticker": "BBB", "decision": "BUY", "score": 1.8, "position_size": 0.2, "decision_reasons": []},
            ]
            out = guard.apply(analyses)
            aaa = [x for x in out if x["ticker"] == "AAA"][0]
            bbb = [x for x in out if x["ticker"] == "BBB"][0]
            self.assertEqual(aaa["decision"], "HOLD")
            self.assertAlmostEqual(float(bbb["position_size"]), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
