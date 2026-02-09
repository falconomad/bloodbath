import unittest
from unittest.mock import patch

import pandas as pd

from src.analysis import dip
from src.api import data_fetcher
from src.core import sp500_list


class DipStrategyTests(unittest.TestCase):
    class YFRateLimitError(Exception):
        pass

    def test_dip_bonus_requires_stabilization_for_full_boost(self):
        # 30% drawdown and still sliding in the latest week -> reduced bonus.
        data = pd.DataFrame({"Close": [100.0, 95.0, 90.0, 88.0, 85.0, 80.0, 75.0, 70.0]})
        bonus, drawdown, stabilized, penalty = dip.dip_bonus(data, dip_threshold=0.2, lookback_days=20)

        self.assertAlmostEqual(drawdown, 0.3)
        self.assertFalse(stabilized)
        self.assertAlmostEqual(bonus, 0.2)
        self.assertEqual(penalty, 0.0)

    def test_dip_bonus_scales_when_bottom_is_stabilizing(self):
        # 30% drawdown but flat/up in recent days around short-term mean.
        data = pd.DataFrame({"Close": [100.0, 95.0, 90.0, 85.0, 80.0, 72.0, 70.0, 70.5, 71.0, 71.5, 72.0]})
        bonus, drawdown, stabilized, _penalty = dip.dip_bonus(data, dip_threshold=0.2, lookback_days=20)

        self.assertAlmostEqual(drawdown, 0.28, places=2)
        self.assertTrue(stabilized)
        self.assertAlmostEqual(bonus, 0.32, places=2)

    def test_volatility_penalty_applies_to_unstable_names(self):
        data = pd.DataFrame({"Close": [100, 80, 120, 75, 125, 70, 130, 68, 135, 65, 140, 62, 145, 60, 150, 58, 155, 56, 160, 54, 165]})
        penalty = dip.volatility_risk_penalty(data)
        self.assertLess(penalty, 0.0)

    @patch("pandas.read_html", side_effect=Exception("offline"))
    def test_sp500_universe_uses_fallback_when_unavailable(self, _mock_read_html):
        fallback = ["AAPL", "MSFT"]
        symbols = sp500_list.get_sp500_universe(fallback=fallback)

        self.assertEqual(symbols, fallback)

    @patch("yfinance.download")
    def test_bulk_price_data_handles_multiindex_download(self, mock_download):
        index = pd.date_range("2024-01-01", periods=2)
        columns = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Close", "Open"]])
        mock_download.return_value = pd.DataFrame(
            [[100, 99, 200, 198], [101, 100, 201, 199]],
            index=index,
            columns=columns,
        )

        result = data_fetcher.get_bulk_price_data(["AAPL", "MSFT"], period="1mo", interval="1d")

        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertFalse(result["AAPL"].empty)
        self.assertFalse(result["MSFT"].empty)

    @patch("src.api.data_fetcher.time.sleep")
    @patch("yfinance.download")
    def test_get_price_data_retries_after_rate_limit(self, mock_download, _mock_sleep):
        index = pd.date_range("2024-01-01", periods=2)
        mock_download.side_effect = [
            self.YFRateLimitError("Too Many Requests. Rate limited."),
            pd.DataFrame({"Close": [100.0, 101.0]}, index=index),
        ]

        result = data_fetcher.get_price_data("AAPL", period="1mo", interval="1d", max_retries=2)

        self.assertFalse(result.empty)
        self.assertEqual(mock_download.call_count, 2)

    @patch("src.api.data_fetcher.get_price_data")
    @patch("yfinance.download")
    def test_bulk_price_data_falls_back_when_bulk_empty(self, mock_download, mock_get_price_data):
        mock_download.return_value = pd.DataFrame()
        index = pd.date_range("2024-01-01", periods=2)
        mock_get_price_data.side_effect = [
            pd.DataFrame({"Close": [100.0, 101.0]}, index=index),
            pd.DataFrame({"Close": [200.0, 201.0]}, index=index),
        ]

        result = data_fetcher.get_bulk_price_data(["AAPL", "MSFT"], period="1mo", interval="1d")

        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertFalse(result["AAPL"].empty)
        self.assertFalse(result["MSFT"].empty)
        self.assertEqual(mock_get_price_data.call_count, 2)


if __name__ == "__main__":
    unittest.main()
