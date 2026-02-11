import unittest

from src.backtesting.portfolio_backtester import evaluate_candidate_portfolio
from src.backtesting.simple_backtester import BacktestExample, ExecutionModel, Signal


class PortfolioBacktesterTests(unittest.TestCase):
    def test_portfolio_backtester_returns_metrics(self):
        examples = [
            BacktestExample(
                ticker="AAA",
                ts="2026-01-01T00:00:00+00:00",
                price=100.0,
                forward_return=0.03,
                signals={
                    "trend": Signal("trend", 0.8, 0.9, True, ""),
                    "sentiment": Signal("sentiment", 0.5, 0.8, True, ""),
                    "events": Signal("events", 0.2, 0.7, True, ""),
                },
                risk_context={"rel_volume": 1.2, "micro_available": True, "data_quality_ok": True, "atr_pct": 0.02},
            ),
            BacktestExample(
                ticker="AAA",
                ts="2026-01-02T00:00:00+00:00",
                price=103.0,
                forward_return=0.02,
                signals={
                    "trend": Signal("trend", 0.7, 0.9, True, ""),
                    "sentiment": Signal("sentiment", 0.4, 0.8, True, ""),
                    "events": Signal("events", 0.1, 0.7, True, ""),
                },
                risk_context={"rel_volume": 1.1, "micro_available": True, "data_quality_ok": True, "atr_pct": 0.02},
            ),
        ]
        cfg = {
            "weights": {"trend": 0.6, "sentiment": 0.3, "events": 0.1},
            "thresholds": {"buy_score": 0.45, "sell_score": -0.45, "min_confidence": 0.4, "force_hold_confidence": 0.3, "high_conflict_force_hold": 0.8},
            "risk": {"conflict_penalty": 0.2, "max_confidence_drop_on_conflict": 0.75, "min_rel_volume_for_buy": 0.5, "extreme_negative_sentiment": -0.8, "min_position_size": 0.02, "max_position_size": 0.2, "volatility_low_atr_pct": 0.02, "volatility_high_atr_pct": 0.08, "volatility_confidence_penalty_max": 0.35, "volatility_position_scale_min": 0.35, "max_data_gap_ratio_for_trade": 0.2, "require_micro_for_buy": False},
            "quality": {"min_usable_signals": 2},
            "stability": {"min_decision_hold_cycles": 1, "min_cycles_between_flips": 1, "min_cycles_between_non_hold_signals": 1},
            "veto": {"enabled": False},
        }
        result = evaluate_candidate_portfolio(
            examples=examples,
            cfg=cfg,
            weights=cfg["weights"],
            execution_model=ExecutionModel(fee_bps=1.0, spread_bps=2.0, slippage_bps=1.0, fill_ratio=1.0),
        )
        self.assertIn("total_return", result)
        self.assertIn("max_drawdown", result)
        self.assertIn("profit_factor", result)


if __name__ == "__main__":
    unittest.main()
