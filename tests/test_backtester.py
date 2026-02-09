import json
import tempfile
import unittest
from pathlib import Path

from src.backtesting.simple_backtester import build_examples, load_trace_entries, tune_from_trace


class SimpleBacktesterTests(unittest.TestCase):
    def _write_trace(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        path = Path(tmp.name)
        try:
            for row in rows:
                tmp.write(json.dumps(row) + "\n")
        finally:
            tmp.close()
        return path

    def test_build_examples_and_tune(self):
        # AAA uptrend, BBB downtrend across two steps each.
        rows = [
            {
                "ts": "2026-01-01T10:00:00+00:00",
                "ticker": "AAA",
                "price": 100.0,
                "signals": {
                    "trend": {"value": 0.8, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": 0.5, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": 0.3, "confidence": 0.7, "quality_ok": True},
                },
                "risk_context": {"rel_volume": 1.2, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02},
            },
            {
                "ts": "2026-01-02T10:00:00+00:00",
                "ticker": "AAA",
                "price": 105.0,
                "signals": {
                    "trend": {"value": 0.7, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": 0.4, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": 0.2, "confidence": 0.7, "quality_ok": True},
                },
                "risk_context": {"rel_volume": 1.1, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02},
            },
            {
                "ts": "2026-01-01T10:00:00+00:00",
                "ticker": "BBB",
                "price": 100.0,
                "signals": {
                    "trend": {"value": -0.8, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": -0.5, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": -0.3, "confidence": 0.7, "quality_ok": True},
                },
                "risk_context": {"rel_volume": 1.2, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02},
            },
            {
                "ts": "2026-01-02T10:00:00+00:00",
                "ticker": "BBB",
                "price": 95.0,
                "signals": {
                    "trend": {"value": -0.7, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": -0.4, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": -0.2, "confidence": 0.7, "quality_ok": True},
                },
                "risk_context": {"rel_volume": 1.1, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02},
            },
        ]
        path = self._write_trace(rows)

        loaded = load_trace_entries(str(path))
        examples = build_examples(loaded, horizon=1)
        self.assertEqual(len(examples), 2)

        result = tune_from_trace(str(path), trials=20, horizon=1, seed=13)
        self.assertEqual(result.get("status"), "ok")
        self.assertEqual(result.get("examples"), 2)
        self.assertIn("best_weights", result)
        self.assertIn("best_metrics", result)

        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
