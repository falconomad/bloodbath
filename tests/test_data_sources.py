import unittest
from unittest.mock import patch
import tempfile

import pandas as pd
from src.api import data_fetcher


class DataSourceTests(unittest.TestCase):
    @patch("src.api.data_fetcher.ENABLE_GOOGLE_NEWS_RSS", True)
    @patch("src.api.data_fetcher._get_google_news_rss")
    @patch("src.api.data_fetcher.client")
    def test_get_company_news_merges_primary_and_google_rss(self, mock_client, mock_rss):
        mock_client.company_news.return_value = [
            {"headline": "Apple beats estimates", "source": "Reuters", "datetime": 1739203200},
            {"headline": "Apple launches new device", "source": "Bloomberg", "datetime": 1739206800},
        ]
        mock_rss.return_value = [
            {"headline": "Apple beats estimates", "source": "Reuters", "datetime": "Mon, 10 Feb 2026 08:00:00 GMT"},
            {"headline": "Apple AI strategy update", "source": "The Verge", "datetime": "Mon, 10 Feb 2026 09:00:00 GMT"},
        ]

        news = data_fetcher.get_company_news("AAPL", structured=True, limit=5)

        self.assertEqual(len(news), 3)
        headlines = [n["headline"] for n in news]
        self.assertIn("Apple beats estimates", headlines)
        self.assertIn("Apple launches new device", headlines)
        self.assertIn("Apple AI strategy update", headlines)

    @patch("src.api.data_fetcher.requests.get")
    def test_google_news_rss_parser_extracts_items(self, mock_get):
        xml = """
        <rss><channel>
          <item>
            <title>Alpha headline</title>
            <source url="https://example.com">Example</source>
            <pubDate>Mon, 10 Feb 2026 09:00:00 GMT</pubDate>
          </item>
          <item>
            <title>Beta headline</title>
            <source url="https://example.org">Example Org</source>
            <pubDate>Mon, 10 Feb 2026 10:00:00 GMT</pubDate>
          </item>
        </channel></rss>
        """
        resp = unittest.mock.Mock()
        resp.raise_for_status.return_value = None
        resp.text = xml
        mock_get.return_value = resp

        items = data_fetcher._get_google_news_rss("AAPL", limit=2)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["headline"], "Alpha headline")
        self.assertEqual(items[1]["source"], "Example Org")

    @patch("src.api.data_fetcher.client")
    def test_get_company_news_structured_payload(self, mock_client):
        mock_client.company_news.return_value = [
            {"headline": "A", "source": "Reuters", "datetime": 1739203200},
            {"headline": "B", "source": "Bloomberg", "datetime": 1739206800},
        ]
        news = data_fetcher.get_company_news("AAPL", structured=True, limit=2)
        self.assertEqual(len(news), 2)
        self.assertTrue(isinstance(news[0], dict))
        self.assertIn("headline", news[0])
        self.assertIn("source", news[0])

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

    @patch("src.api.data_fetcher.ALPACA_API_SECRET", "secret")
    @patch("src.api.data_fetcher.ALPACA_API_KEY", "key")
    @patch("src.api.data_fetcher._get_price_data_from_alpaca")
    @patch("src.api.data_fetcher.client")
    def test_get_price_data_uses_alpaca_first(self, mock_client, mock_alpaca):
        mock_client.stock_candles.return_value = {
            "s": "ok",
            "t": [1704067200, 1704153600],
            "o": [100.0, 101.0],
            "h": [102.0, 103.0],
            "l": [99.0, 100.0],
            "c": [101.0, 102.0],
            "v": [100000, 110000],
        }
        index = pd.date_range("2024-01-01", periods=2)
        mock_alpaca.return_value = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)

        frame = data_fetcher.get_price_data("AAPL", period="1mo", interval="1d")

        self.assertFalse(frame.empty)
        self.assertIn("Close", frame.columns)
        self.assertEqual(len(frame), 2)
        mock_client.stock_candles.assert_not_called()

    @patch("src.api.data_fetcher.FINNHUB_KEY", "demo")
    @patch("src.api.data_fetcher.client")
    @patch("src.api.data_fetcher._get_price_data_from_alpaca")
    def test_get_price_data_falls_back_to_finnhub_when_alpaca_no_data(self, mock_alpaca, mock_client):
        mock_alpaca.return_value = pd.DataFrame()
        mock_client.stock_candles.return_value = {
            "s": "ok",
            "t": [1704067200, 1704153600],
            "o": [100.0, 101.0],
            "h": [102.0, 103.0],
            "l": [99.0, 100.0],
            "c": [101.0, 102.0],
            "v": [100000, 110000],
        }

        with patch("yfinance.download", return_value=pd.DataFrame()):
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

    @patch("src.api.data_fetcher.ALPACA_API_SECRET", "secret")
    @patch("src.api.data_fetcher.ALPACA_API_KEY", "key")
    @patch("src.api.data_fetcher.requests.get")
    def test_alpaca_snapshot_features_parsing(self, mock_requests_get):
        mock_response = unittest.mock.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "latestQuote": {"bp": 100.0, "ap": 100.2},
            "dailyBar": {"o": 99.0, "c": 100.0, "v": 150000},
            "prevDailyBar": {"v": 100000},
        }
        mock_requests_get.return_value = mock_response

        features = data_fetcher.get_alpaca_snapshot_features("AAPL")

        self.assertTrue(features["available"])
        self.assertGreater(features["spread_bps"], 0)
        self.assertGreater(features["rel_volume"], 1.0)
        self.assertGreaterEqual(features["quality"], 0.0)


if __name__ == "__main__":
    unittest.main()
