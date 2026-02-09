import unittest

from src.pipeline.decision_engine import (
    Signal,
    aggregate_confidence,
    decide,
    hard_guardrails,
    normalize_signals,
    weighted_score,
)


class DecisionEngineTests(unittest.TestCase):
    def _cfg(self):
        return {
            "thresholds": {
                "buy_score": 0.45,
                "sell_score": -0.45,
                "min_confidence": 0.45,
                "force_hold_confidence": 0.35,
                "high_conflict_force_hold": 0.7,
            },
            "risk": {
                "extreme_negative_sentiment": -0.7,
                "min_rel_volume_for_buy": 0.75,
                "min_position_size": 0.02,
                "max_position_size": 0.20,
                "max_data_gap_ratio_for_trade": 0.05,
                "require_micro_for_buy": True,
                "conflict_penalty": 0.2,
                "max_confidence_drop_on_conflict": 0.75,
            },
            "quality": {"min_usable_signals": 3},
            "stability": {
                "min_decision_hold_cycles": 1,
                "min_cycles_between_flips": 1,
                "min_cycles_between_non_hold_signals": 1,
            },
        }

    def test_signals_are_normalized(self):
        signals = {
            "trend": Signal("trend", 2.5, 1.4, True),
            "sentiment": Signal("sentiment", -3.0, -0.2, True),
        }
        normalized = normalize_signals(signals)
        self.assertEqual(normalized["trend"].value, 1.0)
        self.assertEqual(normalized["trend"].confidence, 1.0)
        self.assertEqual(normalized["sentiment"].value, -1.0)
        self.assertEqual(normalized["sentiment"].confidence, 0.0)

    def test_weighted_average_ignores_low_quality_signals(self):
        signals = {
            "trend": Signal("trend", 1.0, 0.8, True),
            "sentiment": Signal("sentiment", -1.0, 0.8, False),
        }
        score = weighted_score(signals, {"trend": 0.5, "sentiment": 0.5})
        self.assertEqual(score, 1.0)

    def test_decision_forces_hold_below_confidence(self):
        cfg = self._cfg()
        signals = {
            "trend": Signal("trend", 0.5, 0.8, True),
            "sentiment": Signal("sentiment", 0.2, 0.6, True),
            "events": Signal("events", 0.1, 0.6, True),
        }
        decision, reasons, size = decide(
            ticker="AAA",
            score=0.7,
            confidence=0.2,
            signals=signals,
            risk_context={"rel_volume": 1.0, "data_quality_ok": True, "data_gap_ratio": 0.0, "micro_available": True},
            state={},
            cycle_idx=1,
            cfg=cfg,
        )
        self.assertEqual(decision, "HOLD")
        self.assertIn("confidence:force_hold", reasons)
        self.assertEqual(size, 0.0)

    def test_conflict_reduces_confidence(self):
        signals = {
            "trend": Signal("trend", 1.0, 0.9, True),
            "sentiment": Signal("sentiment", -1.0, 0.9, True),
        }
        conf, conflict = aggregate_confidence(
            signals, 0.2, weights={"trend": 0.5, "sentiment": 0.5}, max_conflict_drop=0.75
        )
        self.assertGreater(conflict, 0.0)
        self.assertLess(conf, 0.9)

    def test_high_conflict_forces_hold(self):
        cfg = self._cfg()
        cfg["quality"]["min_usable_signals"] = 2
        cfg["risk"]["extreme_negative_sentiment"] = -1.1
        signals = {
            "trend": Signal("trend", 1.0, 0.9, True),
            "sentiment": Signal("sentiment", -1.0, 0.9, True),
            "events": Signal("events", 0.0, 0.8, False),
        }
        decision, reasons, _ = decide(
            ticker="AAA",
            score=0.8,
            confidence=0.9,
            signals=signals,
            risk_context={"rel_volume": 1.2, "data_quality_ok": True, "data_gap_ratio": 0.0, "micro_available": True},
            state={},
            cycle_idx=1,
            cfg=cfg,
        )
        self.assertEqual(decision, "HOLD")
        self.assertIn("conflict:high_disagreement", reasons)

    def test_guardrails_block_unsafe_context(self):
        cfg = self._cfg()
        signals = {
            "trend": Signal("trend", 1.0, 0.8, True),
            "sentiment": Signal("sentiment", 0.5, 0.7, False),
            "events": Signal("events", 0.4, 0.6, False),
        }
        reasons = hard_guardrails(
            signals,
            {
                "rel_volume": 0.4,
                "data_quality_ok": False,
                "data_gap_ratio": 0.12,
                "micro_available": False,
            },
            cfg,
        )
        self.assertIn("guardrail:insufficient_usable_signals", reasons)
        self.assertIn("guardrail:price_data_quality", reasons)
        self.assertIn("guardrail:data_gaps", reasons)
        self.assertIn("guardrail:low_volume", reasons)
        self.assertIn("guardrail:micro_unavailable", reasons)

    def test_stability_non_hold_cooldown(self):
        cfg = self._cfg()
        cfg["stability"]["min_cycles_between_non_hold_signals"] = 3
        signals = {
            "trend": Signal("trend", 1.0, 0.9, True),
            "sentiment": Signal("sentiment", 0.6, 0.8, True),
            "events": Signal("events", 0.4, 0.7, True),
        }
        state = {}
        d1, _, _ = decide(
            ticker="AAA",
            score=0.8,
            confidence=0.8,
            signals=signals,
            risk_context={"rel_volume": 1.0, "data_quality_ok": True, "data_gap_ratio": 0.0, "micro_available": True},
            state=state,
            cycle_idx=1,
            cfg=cfg,
        )
        self.assertEqual(d1, "BUY")
        d2, reasons, _ = decide(
            ticker="AAA",
            score=0.8,
            confidence=0.2,
            signals=signals,
            risk_context={"rel_volume": 1.0, "data_quality_ok": True, "data_gap_ratio": 0.0, "micro_available": True},
            state=state,
            cycle_idx=2,
            cfg=cfg,
        )
        self.assertEqual(d2, "HOLD")
        d3, reasons, _ = decide(
            ticker="AAA",
            score=0.8,
            confidence=0.8,
            signals=signals,
            risk_context={"rel_volume": 1.0, "data_quality_ok": True, "data_gap_ratio": 0.0, "micro_available": True},
            state=state,
            cycle_idx=3,
            cfg=cfg,
        )
        self.assertEqual(d3, "HOLD")
        self.assertIn("stability:min_non_hold_gap", reasons)


if __name__ == "__main__":
    unittest.main()
