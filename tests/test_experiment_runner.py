import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.runner import run_experiment


class ExperimentRunnerTests(unittest.TestCase):
    def _write_trace(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        p = Path(tmp.name)
        try:
            for r in rows:
                tmp.write(json.dumps(r) + "\n")
        finally:
            tmp.close()
        return p

    def test_run_experiment_outputs_files(self):
        rows = [
            {
                "ts": "2026-02-01T10:00:00+00:00",
                "ticker": "AAA",
                "price": 100.0,
                "signals": {
                    "trend": {"value": 0.8, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": 0.4, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": 0.2, "confidence": 0.7, "quality_ok": True}
                },
                "risk_context": {"rel_volume": 1.2, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02}
            },
            {
                "ts": "2026-02-02T10:00:00+00:00",
                "ticker": "AAA",
                "price": 104.0,
                "signals": {
                    "trend": {"value": 0.7, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": 0.3, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": 0.2, "confidence": 0.7, "quality_ok": True}
                },
                "risk_context": {"rel_volume": 1.1, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02}
            },
            {
                "ts": "2026-02-01T10:00:00+00:00",
                "ticker": "BBB",
                "price": 100.0,
                "signals": {
                    "trend": {"value": -0.8, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": -0.4, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": -0.2, "confidence": 0.7, "quality_ok": True}
                },
                "risk_context": {"rel_volume": 1.2, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02}
            },
            {
                "ts": "2026-02-02T10:00:00+00:00",
                "ticker": "BBB",
                "price": 96.0,
                "signals": {
                    "trend": {"value": -0.7, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": -0.3, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": -0.2, "confidence": 0.7, "quality_ok": True}
                },
                "risk_context": {"rel_volume": 1.1, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02}
            }
        ]
        trace = self._write_trace(rows)
        cfg_path = Path("tests/fixtures/experiments/basic_experiment.json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        out_path = Path(out.name)
        out.close()

        result = run_experiment(str(trace), cfg, output_path=str(out_path))

        self.assertEqual(result.get("status"), "ok")
        self.assertTrue(Path(result["output_json"]).exists())
        self.assertTrue(Path(result["output_csv"]).exists())
        self.assertTrue(result.get("best_variant") is not None)

        trace.unlink(missing_ok=True)
        Path(result["output_json"]).unlink(missing_ok=True)
        Path(result["output_csv"]).unlink(missing_ok=True)

    def test_run_experiment_with_auto_tune(self):
        rows = [
            {
                "ts": "2026-02-01T10:00:00+00:00",
                "ticker": "AAA",
                "price": 100.0,
                "signals": {
                    "trend": {"value": 0.8, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": 0.4, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": 0.2, "confidence": 0.7, "quality_ok": True},
                },
                "risk_context": {"rel_volume": 1.2, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02},
            },
            {
                "ts": "2026-02-02T10:00:00+00:00",
                "ticker": "AAA",
                "price": 104.0,
                "signals": {
                    "trend": {"value": 0.7, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": 0.3, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": 0.2, "confidence": 0.7, "quality_ok": True},
                },
                "risk_context": {"rel_volume": 1.1, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.02},
            },
        ]
        trace = self._write_trace(rows)
        cfg = {
            "horizon": 1,
            "walk_forward": {"enabled": False},
            "auto_tune": {"enabled": True, "trials": 3, "seed": 5},
        }
        out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        out_path = Path(out.name)
        out.close()
        result = run_experiment(str(trace), cfg, output_path=str(out_path))
        self.assertEqual(result.get("status"), "ok")
        variants = result.get("variants", [])
        self.assertTrue(any(str(v).startswith("auto_") for v in variants))


if __name__ == "__main__":
    unittest.main()
