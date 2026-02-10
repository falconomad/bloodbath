import json
import tempfile
import unittest
from pathlib import Path

from src.backtesting.simple_backtester import (
    ExecutionModel,
    build_examples,
    evaluate_candidate,
    load_trace_entries,
    tune_from_trace,
)


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
        self.assertIn("execution_cost_rate", result["best_metrics"])
        self.assertIn("sharpe_ratio", result["best_metrics"])
        self.assertIn("sortino_ratio", result["best_metrics"])
        self.assertIn("max_drawdown", result["best_metrics"])
        self.assertIn("expectancy_per_trade", result["best_metrics"])
        self.assertIn("turnover", result["best_metrics"])
        self.assertIn("risk_adjusted_objective", result["best_metrics"])

        path.unlink(missing_ok=True)

    def test_execution_costs_reduce_returns(self):
        rows = [
            {
                "ts": "2026-01-01T10:00:00+00:00",
                "ticker": "AAA",
                "price": 100.0,
                "signals": {
                    "trend": {"value": 0.9, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": 0.7, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": 0.4, "confidence": 0.7, "quality_ok": True},
                },
                "risk_context": {"rel_volume": 1.3, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.01},
            },
            {
                "ts": "2026-01-02T10:00:00+00:00",
                "ticker": "AAA",
                "price": 110.0,
                "signals": {
                    "trend": {"value": 0.8, "confidence": 0.9, "quality_ok": True},
                    "sentiment": {"value": 0.6, "confidence": 0.8, "quality_ok": True},
                    "events": {"value": 0.3, "confidence": 0.7, "quality_ok": True},
                },
                "risk_context": {"rel_volume": 1.2, "micro_available": True, "data_quality_ok": True, "data_gap_ratio": 0.0, "atr_pct": 0.01},
            },
        ]
        path = self._write_trace(rows)
        examples = build_examples(load_trace_entries(str(path)), horizon=1)
        cfg = {
            "weights": {"trend": 0.6, "sentiment": 0.3, "events": 0.1},
            "thresholds": {
                "buy_score": 0.45,
                "sell_score": -0.45,
                "min_confidence": 0.45,
                "force_hold_confidence": 0.35,
                "high_conflict_force_hold": 0.7,
            },
            "risk": {
                "conflict_penalty": 0.2,
                "max_confidence_drop_on_conflict": 0.75,
                "min_rel_volume_for_buy": 0.75,
                "extreme_negative_sentiment": -0.7,
                "min_position_size": 0.02,
                "max_position_size": 0.20,
                "volatility_low_atr_pct": 0.02,
                "volatility_high_atr_pct": 0.08,
                "volatility_confidence_penalty_max": 0.35,
                "volatility_position_scale_min": 0.35,
                "max_data_gap_ratio_for_trade": 0.05,
                "require_micro_for_buy": True,
            },
            "quality": {"min_usable_signals": 3},
            "stability": {"min_decision_hold_cycles": 1, "min_cycles_between_flips": 1, "min_cycles_between_non_hold_signals": 1},
            "veto": {
                "enabled": True,
                "block_on_any_guardrail": True,
                "max_conflict_for_trade": 0.7,
                "require_trend_alignment": True,
                "min_quality_signals_for_trade": 3,
                "min_confidence_for_trade": 0.45,
            },
        }
        weights = cfg["weights"]

        no_cost = evaluate_candidate(examples, cfg, weights, execution_model=ExecutionModel(fee_bps=0, spread_bps=0, slippage_bps=0, fill_ratio=1.0))
        with_cost = evaluate_candidate(examples, cfg, weights, execution_model=ExecutionModel(fee_bps=10, spread_bps=20, slippage_bps=10, fill_ratio=0.7))
        self.assertGreater(no_cost["total_return"], with_cost["total_return"])

        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
