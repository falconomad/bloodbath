import unittest

from src.analytics.performance_reports import generate_performance_report


class PerformanceReportTests(unittest.TestCase):
    def test_report_metrics(self):
        # Two tickers over two timestamps, one BUY should win, one SELL should win.
        rows = [
            {
                "ts": "2026-02-01T10:00:00+00:00",
                "ticker": "AAA",
                "decision": "BUY",
                "price": 100.0,
                "signals": {
                    "trend": {"value": 0.7},
                    "sentiment": {"value": 0.4},
                },
            },
            {
                "ts": "2026-02-02T10:00:00+00:00",
                "ticker": "AAA",
                "decision": "HOLD",
                "price": 105.0,
                "signals": {
                    "trend": {"value": 0.6},
                    "sentiment": {"value": 0.3},
                },
            },
            {
                "ts": "2026-02-01T10:00:00+00:00",
                "ticker": "BBB",
                "decision": "SELL",
                "price": 100.0,
                "signals": {
                    "trend": {"value": -0.8},
                    "sentiment": {"value": -0.5},
                },
            },
            {
                "ts": "2026-02-02T10:00:00+00:00",
                "ticker": "BBB",
                "decision": "HOLD",
                "price": 95.0,
                "signals": {
                    "trend": {"value": -0.6},
                    "sentiment": {"value": -0.3},
                },
            },
        ]

        report = generate_performance_report(rows, horizon=1)

        self.assertEqual(report["trades"], 2)
        self.assertEqual(report["wins"], 2)
        self.assertGreater(report["signal_accuracy"], 0.99)
        self.assertGreater(report["profit_factor"], 1.0)
        self.assertIn("module_contributions", report)
        self.assertIn("trend", report["module_contributions"])


if __name__ == "__main__":
    unittest.main()
