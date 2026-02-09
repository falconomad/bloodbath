import unittest
from unittest.mock import patch
import tempfile

import pandas as pd
from src.api import data_fetcher


class DataSourceTests(unittest.TestCase):
    @patch("src.api.data_fetcher.client")
    def test_get_earnings_calendar_uses_finnhub_client(self, mock_client):
        mock_client.earnings_calendar.return_value = {
            "earningsCalendar": [{"symbol": "AAPL", "date": "2026-01-20"}]
        }

        events = data_fetcher.get_earnings_calendar("AAPL", lookahead_days=10)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["symbol"], "AAPL")

    @patch("src.api.data_fetcher.ALPHAVANTAGE_KEY", None)
    @patch("yfinance.download")
    def test_get_price_data_uses_recent_cache_when_live_fetch_fails(self, mock_download):
        mock_download.side_effect = [pd.DataFrame({"Close": [100.0, 101.0]}), pd.DataFrame()]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"PRICE_CACHE_DIR": tmpdir}):
                first = data_fetcher.get_price_data("AAPL", period="1mo", interval="1d", max_retries=1)
                second = data_fetcher.get_price_data("AAPL", period="1mo", interval="1d", max_retries=1)

        self.assertFalse(first.empty)
        self.assertFalse(second.empty)
        self.assertEqual(len(second), 2)


if __name__ == "__main__":
    unittest.main()
