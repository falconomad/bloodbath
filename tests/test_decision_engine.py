import unittest

from src.pipeline.decision_engine import Signal, aggregate_confidence, decide, normalize_signals, weighted_score


class DecisionEngineTests(unittest.TestCase):
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
        cfg = {
            "thresholds": {
                "buy_score": 0.45,
                "sell_score": -0.45,
                "min_confidence": 0.45,
                "force_hold_confidence": 0.35,
            },
            "risk": {
                "extreme_negative_sentiment": -0.7,
                "min_rel_volume_for_buy": 0.75,
                "min_position_size": 0.02,
                "max_position_size": 0.20,
            },
            "stability": {"min_decision_hold_cycles": 1, "min_cycles_between_flips": 1},
        }
        signals = {"sentiment": Signal("sentiment", 0.2, 0.6, True)}
        decision, reasons, size = decide(
            ticker="AAA",
            score=0.7,
            confidence=0.2,
            signals=signals,
            risk_context={"rel_volume": 1.0},
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
        conf, conflict = aggregate_confidence(signals, 0.2)
        self.assertGreater(conflict, 0.0)
        self.assertLess(conf, 0.9)


if __name__ == "__main__":
    unittest.main()
