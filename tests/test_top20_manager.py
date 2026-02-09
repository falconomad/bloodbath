import unittest

from src.core.top20_manager import Top20AutoManager


class Top20ManagerTests(unittest.TestCase):
    def test_execution_costs_reduce_realized_performance(self):
        no_cost = Top20AutoManager(
            starting_capital=1000,
            max_positions=1,
            max_allocation_per_position=1.0,
            slippage_bps=0.0,
            fee_bps=0.0,
        )
        with_cost = Top20AutoManager(
            starting_capital=1000,
            max_positions=1,
            max_allocation_per_position=1.0,
            slippage_bps=20.0,
            fee_bps=10.0,
        )

        buy = [{"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 100.0}]
        sell = [{"ticker": "AAA", "decision": "SELL", "score": -2.0, "price": 110.0}]

        no_cost.step(buy)
        no_cost.step(sell)
        with_cost.step(buy)
        with_cost.step(sell)

        self.assertGreater(no_cost.cash, with_cost.cash)

    def test_allocates_across_multiple_buy_candidates(self):
        manager = Top20AutoManager(starting_capital=500, max_positions=3, max_allocation_per_position=0.5)

        manager.step(
            [
                {"ticker": "AAA", "decision": "BUY", "score": 3.0, "price": 100.0},
                {"ticker": "BBB", "decision": "BUY", "score": 2.5, "price": 50.0},
                {"ticker": "CCC", "decision": "BUY", "score": 2.0, "price": 25.0},
            ]
        )

        self.assertGreaterEqual(len(manager.holdings), 2)
        self.assertIn("AAA", manager.holdings)
        self.assertIn("BBB", manager.holdings)

        total_invested = sum(
            manager.holdings[t]["shares"] * manager.last_price_by_ticker[t] for t in manager.holdings
        )
        self.assertLessEqual(total_invested, 500.000001)
        self.assertGreaterEqual(manager.cash, 0.0)


    def test_supports_fractional_shares_when_cash_is_below_share_price(self):
        manager = Top20AutoManager(starting_capital=200.0, max_positions=1, max_allocation_per_position=1.0)

        manager.step([
            {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 350.0},
        ])

        self.assertIn("AAA", manager.holdings)
        self.assertGreater(float(manager.holdings["AAA"]["shares"]), 0.0)
        self.assertLess(float(manager.holdings["AAA"]["shares"]), 1.0)

    def test_sells_current_holding_on_explicit_sell_signal(self):
        manager = Top20AutoManager(starting_capital=500, max_positions=3, max_allocation_per_position=0.6)

        manager.step(
            [
                {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 100.0},
                {"ticker": "BBB", "decision": "BUY", "score": 1.8, "price": 50.0},
            ]
        )

        aaa_shares = manager.holdings.get("AAA", {}).get("shares", 0)
        self.assertGreater(aaa_shares, 0)

        manager.step(
            [
                {"ticker": "AAA", "decision": "SELL", "score": -2.0, "price": 90.0},
                {"ticker": "BBB", "decision": "HOLD", "score": 0.1, "price": 55.0},
            ]
        )

        self.assertNotIn("AAA", manager.holdings)

        tx = manager.transactions_df()
        sold_aaa = tx[(tx["ticker"] == "AAA") & (tx["action"] == "SELL")]
        self.assertFalse(sold_aaa.empty)

    def test_valuation_uses_held_ticker_prices(self):
        manager = Top20AutoManager(starting_capital=500, max_positions=2, max_allocation_per_position=0.6)

        manager.step(
            [
                {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 100.0},
                {"ticker": "BBB", "decision": "BUY", "score": 1.9, "price": 50.0},
            ]
        )

        manager.step(
            [
                {"ticker": "AAA", "decision": "HOLD", "score": 0.2, "price": 120.0},
                {"ticker": "BBB", "decision": "HOLD", "score": 0.1, "price": 60.0},
            ]
        )

        history = manager.history_df()
        self.assertGreater(float(history.iloc[-1]["Portfolio Value"]), 500.0)

    def test_position_snapshot_includes_allocation_and_pnl(self):
        manager = Top20AutoManager(starting_capital=500, max_positions=2, max_allocation_per_position=0.6)
        manager.step([
            {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 100.0},
            {"ticker": "BBB", "decision": "BUY", "score": 1.9, "price": 50.0},
        ])
        manager.step([
            {"ticker": "AAA", "decision": "HOLD", "score": 0.2, "price": 110.0},
            {"ticker": "BBB", "decision": "HOLD", "score": 0.1, "price": 45.0},
        ])

        snapshot = manager.position_snapshot_df("2026-01-01 10:00:00")

        self.assertFalse(snapshot.empty)
        self.assertTrue({"allocation", "pnl", "pnl_pct"}.issubset(set(snapshot.columns)))
        self.assertAlmostEqual(float(snapshot["allocation"].sum()), 1.0, places=2)

    def test_stop_loss_sells_even_without_explicit_signal(self):
        manager = Top20AutoManager(starting_capital=500, max_positions=2, max_allocation_per_position=0.6)
        manager.step([
            {"ticker": "AAA", "decision": "BUY", "score": 2.2, "price": 100.0},
        ])

        manager.step([
            {"ticker": "AAA", "decision": "HOLD", "score": 0.0, "price": 85.0},
        ])

        self.assertNotIn("AAA", manager.holdings)

    def test_take_profit_sells_when_gain_threshold_hit(self):
        manager = Top20AutoManager(starting_capital=500, max_positions=2, max_allocation_per_position=0.6)
        manager.step([
            {"ticker": "AAA", "decision": "BUY", "score": 2.2, "price": 100.0},
        ])

        manager.step([
            {"ticker": "AAA", "decision": "HOLD", "score": 0.0, "price": 132.0},
        ])

        self.assertNotIn("AAA", manager.holdings)

    def test_cooldown_prevents_immediate_reentry_after_sell(self):
        manager = Top20AutoManager(
            starting_capital=500,
            max_positions=2,
            max_allocation_per_position=0.6,
            cooldown_after_sell_steps=1,
        )
        manager.step([
            {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 100.0},
        ])
        manager.step([
            {"ticker": "AAA", "decision": "SELL", "score": -2.0, "price": 90.0},
        ])
        self.assertNotIn("AAA", manager.holdings)

        # Immediate next step BUY should be blocked by cooldown.
        manager.step([
            {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 92.0},
        ])
        self.assertNotIn("AAA", manager.holdings)

        # Re-entry allowed once cooldown expires.
        manager.step([
            {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 94.0},
        ])
        self.assertIn("AAA", manager.holdings)

    def test_allocates_more_to_higher_weighted_buy_candidates(self):
        manager = Top20AutoManager(starting_capital=1000, max_positions=3, max_allocation_per_position=0.8)

        manager.step(
            [
                {
                    "ticker": "AAA",
                    "decision": "BUY",
                    "score": 2.0,
                    "price": 100.0,
                    "sentiment": 0.8,
                    "growth_20d": 0.12,
                },
                {
                    "ticker": "BBB",
                    "decision": "BUY",
                    "score": 1.2,
                    "price": 100.0,
                    "sentiment": -0.2,
                    "growth_20d": -0.03,
                },
            ]
        )

        aaa_value = manager.holdings["AAA"]["shares"] * manager.last_price_by_ticker["AAA"]
        bbb_value = manager.holdings["BBB"]["shares"] * manager.last_price_by_ticker["BBB"]
        self.assertGreater(aaa_value, bbb_value)


if __name__ == "__main__":
    unittest.main()
