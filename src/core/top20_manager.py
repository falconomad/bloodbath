import datetime
import pandas as pd


class Top20AutoManager:
    def __init__(self, starting_capital=500, max_positions=5, max_allocation_per_position=0.35):
        self.cash = float(starting_capital)
        self.max_positions = int(max_positions)
        self.max_allocation_per_position = float(max_allocation_per_position)
        self.holdings = {}
        self.last_price_by_ticker = {}
        self.history = []
        self.transactions = []

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

    def _buy(self, timestamp, ticker, price, budget):
        if budget <= 0 or self.cash < price:
            return

        shares = int(min(budget, self.cash) // price)
        if shares <= 0:
            return

        cost = shares * price
        self.cash -= cost
        self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
        self._record(timestamp, ticker, "BUY", shares, price)

    def _sell_all(self, timestamp, ticker, price):
        shares = int(self.holdings.get(ticker, 0))
        if shares <= 0:
            return

        self.cash += shares * price
        self._record(timestamp, ticker, "SELL", shares, price)
        self.holdings.pop(ticker, None)

    def _portfolio_value(self):
        total = self.cash
        for ticker, shares in self.holdings.items():
            price = self.last_price_by_ticker.get(ticker)
            if price is not None:
                total += shares * price
        return round(float(total), 4)

    def step(self, analyses):
        if not analyses:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for analysis in analyses:
            self.last_price_by_ticker[analysis["ticker"]] = float(analysis["price"])

        analysis_by_ticker = {a["ticker"]: a for a in analyses}

        # 1) Risk-first: close any currently held ticker with explicit SELL.
        for ticker in list(self.holdings.keys()):
            rec = analysis_by_ticker.get(ticker)
            if rec and rec["decision"] == "SELL":
                self._sell_all(timestamp, ticker, float(rec["price"]))

        # 2) Open new BUY positions across strongest candidates.
        buy_candidates = [
            a for a in analyses if a["decision"] == "BUY" and float(a.get("score", 0.0)) > 0
        ]
        buy_candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        open_slots = max(self.max_positions - len(self.holdings), 0)
        new_candidates = [c for c in buy_candidates if c["ticker"] not in self.holdings][:open_slots]

        if new_candidates and self.cash > 0:
            equity = self._portfolio_value()
            cap_per_position = equity * self.max_allocation_per_position

            for idx, candidate in enumerate(new_candidates):
                remaining = len(new_candidates) - idx
                budget_split = self.cash / max(remaining, 1)
                budget = min(budget_split, cap_per_position)
                self._buy(
                    timestamp,
                    ticker=candidate["ticker"],
                    price=float(candidate["price"]),
                    budget=budget,
                )

        self.history.append(self._portfolio_value())

    def history_df(self):
        return pd.DataFrame(
            {"Step": list(range(1, len(self.history) + 1)), "Portfolio Value": self.history}
        )

    def transactions_df(self):
        return pd.DataFrame(self.transactions)
