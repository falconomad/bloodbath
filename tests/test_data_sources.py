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

    @patch("src.api.data_fetcher.FINNHUB_KEY", "demo")
    @patch("src.api.data_fetcher.client")
    def test_get_price_data_uses_finnhub_candles_first(self, mock_client):
        mock_client.stock_candles.return_value = {
            "s": "ok",
            "t": [1704067200, 1704153600],
            "o": [100.0, 101.0],
            "h": [102.0, 103.0],
            "l": [99.0, 100.0],
            "c": [101.0, 102.0],
            "v": [100000, 110000],
        }

        frame = data_fetcher.get_price_data("AAPL", period="1mo", interval="1d")

        self.assertFalse(frame.empty)
        self.assertIn("Close", frame.columns)
        self.assertEqual(len(frame), 2)

    @patch("src.api.data_fetcher.FINNHUB_KEY", "demo")
    @patch("src.api.data_fetcher.client")
    def test_get_price_data_falls_back_when_finnhub_no_data(self, mock_client):
        mock_client.stock_candles.return_value = {"s": "no_data"}

        with patch("src.api.data_fetcher._get_price_data_from_alpaca", return_value=pd.DataFrame({"Close": [1.0]})):
            frame = data_fetcher.get_price_data("AAPL", period="1mo", interval="1d")

        self.assertFalse(frame.empty)

    @patch("src.api.data_fetcher.ALPACA_API_SECRET", "secret")
    @patch("src.api.data_fetcher.ALPACA_API_KEY", "key")
    @patch("src.api.data_fetcher.FINNHUB_KEY", "demo")
    @patch("src.api.data_fetcher.client")
    @patch("src.api.data_fetcher.requests.get")
    def test_get_price_data_uses_alpaca_after_finnhub_no_data(self, mock_requests_get, mock_client):
        mock_client.stock_candles.return_value = {"s": "no_data"}
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "bars": [
                {"t": "2026-02-06T00:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 100000},
                {"t": "2026-02-07T00:00:00Z", "o": 101.0, "h": 103.0, "l": 100.0, "c": 102.0, "v": 110000},
            ]
        }
        mock_requests_get.return_value = mock_response

        frame = data_fetcher.get_price_data("AAPL", period="1mo", interval="1d")

        self.assertFalse(frame.empty)
        self.assertIn("Close", frame.columns)
        self.assertEqual(len(frame), 2)

    @patch("src.api.data_fetcher.ALPACA_API_SECRET", "secret")
    @patch("src.api.data_fetcher.ALPACA_API_KEY", "key")
    @patch("src.api.data_fetcher.requests.get")
    def test_alpaca_symbol_conversion_for_share_class(self, mock_requests_get):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "bars": [{"t": "2026-02-07T00:00:00Z", "o": 500.0, "h": 505.0, "l": 495.0, "c": 503.0, "v": 9000}]
        }
        mock_requests_get.return_value = mock_response

        data_fetcher._get_price_data_from_alpaca("BRK-B", period="1mo", interval="1d")

        called_url = mock_requests_get.call_args.args[0]
        self.assertIn("/stocks/BRK.B/bars", called_url)


if __name__ == "__main__":
    unittest.main()
