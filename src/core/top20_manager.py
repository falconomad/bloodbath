import datetime
import pandas as pd


class Top20AutoManager:
    def __init__(self, starting_capital=500):
        self.cash = starting_capital
        self.position = None
        self.shares = 0
        self.last_price_by_ticker = {}
        self.history = []
        self.transactions = []

    def choose_best(self, analyses):
        actionable = [a for a in analyses if a["decision"] in {"BUY", "SELL"}]
        if not actionable:
            return None
        return max(actionable, key=lambda x: x["score"])

    def _record(self, timestamp, ticker, action, shares, price):
        self.transactions.append(
            {
                "time": timestamp,
                "ticker": ticker,
                "action": action,
                "shares": int(shares),
                "price": round(float(price), 2),
            }
        )

    def _sell_position(self, timestamp, price):
        if not self.position or self.shares <= 0:
            return

        held_ticker = self.position
        self.cash += self.shares * price
        self._record(timestamp, held_ticker, "SELL", self.shares, price)
        self.position = None
        self.shares = 0

    def _buy_position(self, timestamp, ticker, price):
        if self.cash < price:
            return

        shares = int(self.cash // price)
        if shares <= 0:
            return

        self.cash -= shares * price
        self.shares = shares
        self.position = ticker
        self._record(timestamp, ticker, "BUY", shares, price)

    def step(self, analyses):
        if not analyses:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for analysis in analyses:
            self.last_price_by_ticker[analysis["ticker"]] = float(analysis["price"])

        best = self.choose_best(analyses)
        if best:
            ticker = best["ticker"]
            decision = best["decision"]
            price = float(best["price"])

            if decision == "SELL" and self.position == ticker:
                self._sell_position(timestamp, price)

            elif decision == "BUY":
                if self.position and self.position != ticker:
                    held_price = self.last_price_by_ticker.get(self.position)
                    if held_price is not None:
                        self._sell_position(timestamp, held_price)

                if self.position is None:
                    self._buy_position(timestamp, ticker, price)

        portfolio_value = self.cash
        if self.position:
            held_price = self.last_price_by_ticker.get(self.position)
            if held_price is not None:
                portfolio_value += self.shares * held_price

        self.history.append(round(float(portfolio_value), 4))

    def history_df(self):
        return pd.DataFrame(
            {"Step": list(range(1, len(self.history) + 1)), "Portfolio Value": self.history}
        )

    def transactions_df(self):
        return pd.DataFrame(self.transactions)
