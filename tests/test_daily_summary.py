import json
import tempfile
import unittest
from pathlib import Path

from src.analytics.daily_summary import generate_daily_summary


class DailySummaryTests(unittest.TestCase):
    def _write_trace(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        p = Path(tmp.name)
        try:
            for r in rows:
                tmp.write(json.dumps(r) + "\n")
        finally:
            tmp.close()
        return p

    def test_generate_daily_summary_outputs_files(self):
        rows = [
            {
                "ts": "2026-02-10T10:00:00+00:00",
                "ticker": "AAA",
                "price": 100.0,
                "decision": "BUY",
                "confidence": 0.7,
                "decision_reasons": ["volatility:low"],
                "signals": {"trend": {"quality_ok": True}, "sentiment": {"quality_ok": True}},
                "risk_context": {"market_regime": "bull"},
            },
            {
                "ts": "2026-02-10T10:30:00+00:00",
                "ticker": "AAA",
                "price": 101.0,
                "decision": "HOLD",
                "confidence": 0.6,
                "decision_reasons": ["guardrail:low_volume"],
                "signals": {"trend": {"quality_ok": True}, "sentiment": {"quality_ok": False}},
                "risk_context": {"market_regime": "bull"},
            },
        ]
        trace = self._write_trace(rows)
        out_dir = tempfile.mkdtemp(prefix="bb_reports_")
        result = generate_daily_summary(trace_path=str(trace), output_dir=out_dir)
        self.assertEqual(result["day"], "2026-02-10")
        self.assertTrue(Path(result["output_json"]).exists())
        self.assertTrue(Path(result["output_csv"]).exists())
        self.assertIn("signal_quality_ratio", result)


if __name__ == "__main__":
    unittest.main()

