import unittest

from src.core.top20_manager import Top20AutoManager


class Top20ManagerTests(unittest.TestCase):
    def test_rebalances_to_better_buy_candidate(self):
        manager = Top20AutoManager(starting_capital=500)

        manager.step(
            [
                {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 100.0},
                {"ticker": "BBB", "decision": "HOLD", "score": 0.2, "price": 50.0},
            ]
        )

        self.assertEqual(manager.position, "AAA")
        self.assertEqual(manager.shares, 5)

        manager.step(
            [
                {"ticker": "AAA", "decision": "HOLD", "score": 0.1, "price": 110.0},
                {"ticker": "BBB", "decision": "BUY", "score": 3.0, "price": 50.0},
            ]
        )

        # Must sell AAA first (5 * 110 = 550), then buy BBB (11 shares @ 50)
        self.assertEqual(manager.position, "BBB")
        self.assertEqual(manager.shares, 11)
        self.assertEqual(manager.cash, 0)

        tx = manager.transactions_df()
        self.assertEqual(len(tx), 3)
        self.assertEqual(tx.iloc[1]["action"], "SELL")
        self.assertEqual(tx.iloc[1]["ticker"], "AAA")
        self.assertEqual(tx.iloc[2]["action"], "BUY")
        self.assertEqual(tx.iloc[2]["ticker"], "BBB")

    def test_valuation_uses_held_ticker_price(self):
        manager = Top20AutoManager(starting_capital=500)

        manager.step(
            [
                {"ticker": "AAA", "decision": "BUY", "score": 2.0, "price": 100.0},
                {"ticker": "BBB", "decision": "HOLD", "score": 0.5, "price": 50.0},
            ]
        )

        manager.step(
            [
                {"ticker": "AAA", "decision": "HOLD", "score": 0.0, "price": 120.0},
                {"ticker": "BBB", "decision": "HOLD", "score": 0.0, "price": 60.0},
            ]
        )

        # Should value using AAA price (5 * 120), not BBB price.
        history = manager.history_df()
        self.assertEqual(float(history.iloc[-1]["Portfolio Value"]), 600.0)


if __name__ == "__main__":
    unittest.main()
