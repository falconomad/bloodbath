import datetime
import pandas as pd


class Top20AutoManager:
    def __init__(self, starting_capital=500, max_positions=5, max_allocation_per_position=0.35):
        self.cash = float(starting_capital)
        self.max_positions = int(max_positions)
        self.max_allocation_per_position = float(max_allocation_per_position)
        # ticker -> {"shares": int, "avg_cost": float}
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

        position = self.holdings.get(ticker, {"shares": 0, "avg_cost": 0.0})
        prev_shares = int(position["shares"])
        prev_cost_basis = prev_shares * float(position["avg_cost"])
        new_shares = prev_shares + shares
        new_avg_cost = (prev_cost_basis + cost) / new_shares

        self.holdings[ticker] = {"shares": new_shares, "avg_cost": new_avg_cost}
        self._record(timestamp, ticker, "BUY", shares, price)

    def _sell_all(self, timestamp, ticker, price):
        position = self.holdings.get(ticker, {"shares": 0})
        shares = int(position["shares"])
        if shares <= 0:
            return

        self.cash += shares * price
        self._record(timestamp, ticker, "SELL", shares, price)
        self.holdings.pop(ticker, None)

    def _portfolio_value(self):
        total = self.cash
        for ticker, position in self.holdings.items():
            price = self.last_price_by_ticker.get(ticker)
            if price is not None:
                total += int(position["shares"]) * price
        return round(float(total), 4)

    def position_snapshot_df(self, timestamp=None):
        if not self.holdings:
            return pd.DataFrame(
                columns=[
                    "time",
                    "ticker",
                    "shares",
                    "avg_cost",
                    "current_price",
                    "market_value",
                    "allocation",
                    "pnl",
                    "pnl_pct",
                ]
            )

        snapshot_time = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        equity = self._portfolio_value()
        rows = []
        for ticker, position in self.holdings.items():
            shares = int(position["shares"])
            avg_cost = float(position["avg_cost"])
            current_price = float(self.last_price_by_ticker.get(ticker, avg_cost))
            market_value = shares * current_price
            cost_basis = shares * avg_cost
            pnl = market_value - cost_basis
            allocation = (market_value / equity) if equity > 0 else 0.0
            pnl_pct = (pnl / cost_basis) if cost_basis > 0 else 0.0

            rows.append(
                {
                    "time": snapshot_time,
                    "ticker": ticker,
                    "shares": shares,
                    "avg_cost": round(avg_cost, 4),
                    "current_price": round(current_price, 4),
                    "market_value": round(market_value, 4),
                    "allocation": round(allocation, 6),
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl_pct, 6),
                }
            )

        return pd.DataFrame(rows).sort_values("market_value", ascending=False)

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
