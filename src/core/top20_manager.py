import datetime
import math
import pandas as pd


class Top20AutoManager:
    def __init__(
        self,
        starting_capital=500,
        max_positions=5,
        max_allocation_per_position=0.35,
        stop_loss_pct=0.12,
        take_profit_pct=0.30,
    ):
        self.cash = float(starting_capital)
        self.max_positions = int(max_positions)
        self.max_allocation_per_position = float(max_allocation_per_position)
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        # ticker -> {"shares": float, "avg_cost": float, "peak_price": float}
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
                "shares": round(float(shares), 6),
                "price": round(float(price), 2),
            }
        )

    def _is_valid_price(self, price):
        return price is not None and math.isfinite(float(price)) and float(price) > 0

    def _buy(self, timestamp, ticker, price, budget):
        if not self._is_valid_price(price) or budget <= 0 or self.cash <= 0:
            return

        shares = min(budget, self.cash) / price
        if shares <= 0 or shares < 1e-6:
            return

        cost = shares * price
        self.cash -= cost

        position = self.holdings.get(ticker, {"shares": 0, "avg_cost": 0.0, "peak_price": float(price)})
        prev_shares = float(position["shares"])
        prev_cost_basis = prev_shares * float(position["avg_cost"])
        new_shares = prev_shares + shares
        new_avg_cost = (prev_cost_basis + cost) / new_shares
        new_peak = max(float(position.get("peak_price", price)), float(price))

        self.holdings[ticker] = {
            "shares": new_shares,
            "avg_cost": new_avg_cost,
            "peak_price": new_peak,
        }
        self._record(timestamp, ticker, "BUY", shares, price)

    def _sell_all(self, timestamp, ticker, price):
        if not self._is_valid_price(price):
            return

        position = self.holdings.get(ticker, {"shares": 0})
        shares = float(position["shares"])
        if shares <= 0 or shares < 1e-6:
            return

        self.cash += shares * price
        self._record(timestamp, ticker, "SELL", shares, price)
        self.holdings.pop(ticker, None)

    def _portfolio_value(self):
        total = self.cash
        for ticker, position in self.holdings.items():
            price = self.last_price_by_ticker.get(ticker)
            if self._is_valid_price(price):
                total += float(position["shares"]) * price
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
            shares = float(position["shares"])
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
            self.history.append(self._portfolio_value())
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for analysis in analyses:
            price = float(analysis["price"])
            if self._is_valid_price(price):
                self.last_price_by_ticker[analysis["ticker"]] = price

        analysis_by_ticker = {a["ticker"]: a for a in analyses}

        # 1) Risk-first exits: explicit SELL, stop-loss, and take-profit.
        for ticker in list(self.holdings.keys()):
            rec = analysis_by_ticker.get(ticker, {})
            price = float(self.last_price_by_ticker.get(ticker, self.holdings[ticker]["avg_cost"]))
            position = self.holdings[ticker]
            avg_cost = float(position["avg_cost"])

            if self._is_valid_price(price):
                position["peak_price"] = max(float(position.get("peak_price", price)), price)

            stop_loss_triggered = price <= avg_cost * (1 - self.stop_loss_pct)
            take_profit_triggered = price >= avg_cost * (1 + self.take_profit_pct)
            explicit_sell = rec.get("decision") == "SELL"

            if explicit_sell or stop_loss_triggered or take_profit_triggered:
                self._sell_all(timestamp, ticker, price)

        # 2) Open new BUY positions across strongest candidates.
        buy_candidates = [
            a
            for a in analyses
            if a["decision"] == "BUY" and float(a.get("score", 0.0)) > 0 and self._is_valid_price(a["price"])
        ]
        buy_candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        open_slots = max(self.max_positions - len(self.holdings), 0)
        new_candidates = [c for c in buy_candidates if c["ticker"] not in self.holdings][:open_slots]

        if self.cash > 0:
            equity = self._portfolio_value()
            cap_per_position = equity * self.max_allocation_per_position

            for idx, candidate in enumerate(new_candidates):
                remaining = len(new_candidates) - idx
                budget_split = self.cash / max(remaining, 1)
                budget = min(budget_split, cap_per_position)
                self._buy(timestamp, ticker=candidate["ticker"], price=float(candidate["price"]), budget=budget)

            # 3) Opportunistic averaging-up for strong signals if under-allocated.
            for candidate in buy_candidates:
                ticker = candidate["ticker"]
                if ticker not in self.holdings:
                    continue
                if float(candidate.get("score", 0.0)) < 1.5:
                    continue

                price = float(candidate["price"])
                market_value = self.holdings[ticker]["shares"] * price
                room = max(cap_per_position - market_value, 0.0)
                if room <= 0:
                    continue
                budget = min(room, self.cash * 0.35)
                self._buy(timestamp, ticker=ticker, price=price, budget=budget)

        self.history.append(self._portfolio_value())

    def history_df(self):
        return pd.DataFrame(
            {"Step": list(range(1, len(self.history) + 1)), "Portfolio Value": self.history}
        )

    def transactions_df(self):
        return pd.DataFrame(self.transactions)
