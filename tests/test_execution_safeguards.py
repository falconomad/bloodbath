import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.execution.safeguards import ExecutionSafeguard


class ExecutionSafeguardTests(unittest.TestCase):
    def test_circuit_breaker_blocks_new_buys_after_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "guard_state.json"
            guard = ExecutionSafeguard(state_path=str(state), max_consecutive_failures=2)
            guard.record_failure("x")
            guard.record_failure("y")
            analyses = [{"ticker": "AAA", "decision": "BUY", "position_size": 0.2}]
            decision = guard.assess(analyses=analyses, candidates={}, portfolio_value=1000.0)
            self.assertTrue(decision.block_new_buys)
            self.assertEqual(decision.filtered_analyses[0]["decision"], "HOLD")
            self.assertTrue(any("circuit_breaker" in r for r in decision.reasons))

    def test_stale_data_blocks_buys(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "guard_state.json"
            guard = ExecutionSafeguard(state_path=str(state), stale_data_max_age_hours=1)
            idx = pd.date_range("2026-01-01", periods=3, freq="D")
            data = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)
            analyses = [{"ticker": "AAA", "decision": "BUY", "position_size": 0.1}]
            decision = guard.assess(analyses=analyses, candidates={"AAA": {"data": data}}, portfolio_value=1000.0)
            self.assertTrue(decision.block_new_buys)
            self.assertTrue(any("stale_data" in r for r in decision.reasons))


if __name__ == "__main__":
    unittest.main()
