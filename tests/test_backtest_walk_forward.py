import unittest

import pandas as pd

from src.backtest.walk_forward import run_walk_forward_backtest


class WalkForwardBacktestTests(unittest.TestCase):
    def test_backtest_returns_metrics_and_history(self):
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        uptrend = pd.DataFrame({"Close": [100 + i * 0.5 for i in range(200)]}, index=dates)
        mild = pd.DataFrame({"Close": [80 + i * 0.2 for i in range(200)]}, index=dates)
        price_map = {"AAA": uptrend, "BBB": mild}

        result = run_walk_forward_backtest(price_map, starting_capital=1000, warmup_bars=60)

        self.assertFalse(result.history.empty)
        self.assertIn("total_return_pct", result.metrics)
        self.assertIn("benchmark_return_pct", result.metrics)
        self.assertIn("excess_return_pct", result.metrics)
        self.assertIn("max_drawdown_pct", result.metrics)
        self.assertGreaterEqual(result.metrics["num_cycles"], 1)

    def test_backtest_handles_empty_input(self):
        result = run_walk_forward_backtest({}, starting_capital=1000, warmup_bars=60)

        self.assertTrue(result.history.empty)
        self.assertEqual(result.metrics["ending_value"], 1000.0)
        self.assertEqual(result.metrics["num_trades"], 0)


if __name__ == "__main__":
    unittest.main()
