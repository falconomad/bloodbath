import unittest

import pandas as pd

from src.backtest.sweep import run_parameter_sweep


class BacktestSweepTests(unittest.TestCase):
    def test_sweep_returns_ranked_rows(self):
        dates = pd.date_range("2024-01-01", periods=220, freq="D")
        uptrend = pd.DataFrame({"Close": [100 + i * 0.4 for i in range(220)]}, index=dates)
        noisy = pd.DataFrame({"Close": [80 + (i * 0.1) + ((-1) ** i) for i in range(220)]}, index=dates)
        price_map = {"AAA": uptrend, "BBB": noisy}

        table = run_parameter_sweep(
            price_map,
            buy_thresholds=[0.55, 1.0],
            sell_thresholds=[-0.7, -1.0],
            min_buy_scores=[0.45, 0.75],
            slippage_bps_values=[5.0],
            fee_bps_values=[1.0],
            max_drawdown_limit_pct=-80.0,
            top_k=5,
        )

        self.assertFalse(table.empty)
        self.assertIn("excess_return_pct", table.columns)
        self.assertLessEqual(len(table), 5)


if __name__ == "__main__":
    unittest.main()
