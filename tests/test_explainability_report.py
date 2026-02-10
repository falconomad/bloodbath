import unittest

from src.analytics.explainability_report import generate_explainability_report


class ExplainabilityReportTests(unittest.TestCase):
    def test_report_contains_reason_breakdown(self):
        rows = [
            {
                "ts": "2026-02-01T10:00:00+00:00",
                "ticker": "AAA",
                "decision": "HOLD",
                "score": 0.2,
                "confidence": 0.3,
                "decision_reasons": ["confidence:force_hold", "veto:low_confidence"],
                "weights": {"trend": 0.5, "sentiment": 0.5},
                "signals": {
                    "trend": {"value": 0.4},
                    "sentiment": {"value": 0.1},
                },
            },
            {
                "ts": "2026-02-01T10:30:00+00:00",
                "ticker": "BBB",
                "decision": "BUY",
                "score": 0.7,
                "confidence": 0.8,
                "decision_reasons": [],
                "weights": {"trend": 0.6, "sentiment": 0.4},
                "signals": {
                    "trend": {"value": 0.8},
                    "sentiment": {"value": 0.5},
                },
            },
        ]

        report = generate_explainability_report(rows, max_examples=5)
        self.assertEqual(report["total_entries"], 2)
        self.assertIn("HOLD", report["decision_counts"])
        self.assertTrue(len(report["top_vetoes"]) >= 1)
        self.assertTrue(len(report["examples"]) == 2)


if __name__ == "__main__":
    unittest.main()
