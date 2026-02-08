import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
