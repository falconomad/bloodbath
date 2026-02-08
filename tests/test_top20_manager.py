import unittest

from src.core.top20_manager import Top20AutoManager


class Top20ManagerTests(unittest.TestCase):
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
            manager.holdings[t] * manager.last_price_by_ticker[t] for t in manager.holdings
        )
        self.assertLessEqual(total_invested, 500.0)
        self.assertGreaterEqual(manager.cash, 0.0)

    def test_sells_current_holding_on_explicit_sell_signal(self):
        manager = Top20AutoManager(starting_capital=500, max_positions=3, max_allocation_per_position=0.6)

        manager.step(
            [
                {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 100.0},
                {"ticker": "BBB", "decision": "BUY", "score": 1.8, "price": 50.0},
            ]
        )

        aaa_shares = manager.holdings.get("AAA", 0)
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


if __name__ == "__main__":
    unittest.main()
