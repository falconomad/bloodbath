import tempfile
import unittest
from pathlib import Path

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


class GeminiGuardTests(unittest.TestCase):
    def test_blocks_top_buy_only_with_cycle_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = _FakeGeminiGuard(
                api_key="x",
                max_calls_per_cycle=1,
                max_calls_per_day=5,
                state_path=str(Path(tmp) / "gemini_state.json"),
            )
            guard.queue({"action": "BLOCK", "confidence": 0.9, "reason": "high risk", "size_mult": 1.0})
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
            guard.queue({"action": "REDUCE", "confidence": 0.7, "reason": "vol", "size_mult": 0.4})
            analyses = [{"ticker": "AAA", "decision": "BUY", "score": 1.1, "position_size": 0.2, "decision_reasons": []}]
            out = guard.apply(analyses)
            self.assertAlmostEqual(float(out[0]["position_size"]), 0.08, places=6)


if __name__ == "__main__":
    unittest.main()
