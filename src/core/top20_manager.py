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
        min_buy_score=0.75,
        cooldown_after_sell_steps=1,
        max_new_exposure_per_step=1.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        sell_confirm_steps=2,
        take_profit_partial_ratio=0.5,
        trailing_atr_mult=2.5,
    ):
        self.cash = float(starting_capital)
        self.max_positions = int(max_positions)
        self.max_allocation_per_position = float(max_allocation_per_position)
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.min_buy_score = float(min_buy_score)
        self.cooldown_after_sell_steps = int(cooldown_after_sell_steps)
        self.max_new_exposure_per_step = float(max_new_exposure_per_step)
        self.slippage_bps = float(slippage_bps)
        self.fee_bps = float(fee_bps)
        self.sell_confirm_steps = int(sell_confirm_steps)
        self.take_profit_partial_ratio = float(take_profit_partial_ratio)
        self.trailing_atr_mult = float(trailing_atr_mult)
        # ticker -> {"shares": float, "avg_cost": float, "peak_price": float}
        self.holdings = {}
        self.last_price_by_ticker = {}
        self.last_sell_step_by_ticker = {}
        self.sell_streak_by_ticker = {}
        self.step_index = 0
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

    def _buy_weight(self, candidate):
        score = max(float(candidate.get("score", 0.0)), 0.0)
        sentiment = float(candidate.get("sentiment", 0.0))
        growth = float(candidate.get("growth_20d", 0.0))

        # Blend conviction (score), sentiment, and recent growth into a positive weight.
        sentiment_factor = max(0.6, min(1.4, 1.0 + (0.30 * sentiment)))
        growth_factor = max(0.7, min(1.5, 1.0 + (0.80 * growth)))
        return max(score * sentiment_factor * growth_factor, 1e-6)

    def _buy(self, timestamp, ticker, price, budget):
        if not self._is_valid_price(price) or budget <= 0 or self.cash <= 0:
            return 0.0

        exec_price = float(price) * (1.0 + (self.slippage_bps / 10_000.0))
        shares = min(budget, self.cash) / exec_price
        if shares <= 0 or shares < 1e-6:
            return 0.0

        notional = shares * exec_price
        fee = notional * (self.fee_bps / 10_000.0)
        total_cost = notional + fee
        self.cash -= total_cost

        position = self.holdings.get(ticker, {"shares": 0, "avg_cost": 0.0, "peak_price": float(price)})
        prev_shares = float(position["shares"])
        prev_cost_basis = prev_shares * float(position["avg_cost"])
        new_shares = prev_shares + shares
        new_avg_cost = (prev_cost_basis + total_cost) / new_shares
        new_peak = max(float(position.get("peak_price", exec_price)), float(exec_price))

        self.holdings[ticker] = {
            "shares": new_shares,
            "avg_cost": new_avg_cost,
            "peak_price": new_peak,
        }
        self._record(timestamp, ticker, "BUY", shares, exec_price)
        return total_cost

    def _sell_all(self, timestamp, ticker, price):
        if not self._is_valid_price(price):
            return

        position = self.holdings.get(ticker, {"shares": 0})
        shares = float(position["shares"])
        if shares <= 0 or shares < 1e-6:
            return

        exec_price = float(price) * (1.0 - (self.slippage_bps / 10_000.0))
        proceeds = shares * exec_price
        fee = proceeds * (self.fee_bps / 10_000.0)
        net_proceeds = proceeds - fee
        self.cash += net_proceeds
        self._record(timestamp, ticker, "SELL", shares, exec_price)
        self.holdings.pop(ticker, None)
        self.last_sell_step_by_ticker[ticker] = self.step_index
        self.sell_streak_by_ticker[ticker] = 0

    def _sell_fraction(self, timestamp, ticker, price, fraction, action="SELL_PARTIAL"):
        if not self._is_valid_price(price):
            return
        if fraction <= 0 or fraction >= 1:
            return

        position = self.holdings.get(ticker, {"shares": 0})
        shares = float(position.get("shares", 0))
        if shares <= 0 or shares < 1e-6:
            return

        sell_shares = shares * fraction
        if sell_shares <= 0 or sell_shares < 1e-6:
            return

        exec_price = float(price) * (1.0 - (self.slippage_bps / 10_000.0))
        proceeds = sell_shares * exec_price
        fee = proceeds * (self.fee_bps / 10_000.0)
        self.cash += proceeds - fee

        remaining = shares - sell_shares
        if remaining <= 1e-6:
            self._record(timestamp, ticker, "SELL", shares, exec_price)
            self.holdings.pop(ticker, None)
            self.last_sell_step_by_ticker[ticker] = self.step_index
            self.sell_streak_by_ticker[ticker] = 0
            return

        self.holdings[ticker]["shares"] = remaining
        self._record(timestamp, ticker, action, sell_shares, exec_price)

    def _portfolio_value(self):
        total = self.cash
        for ticker, position in self.holdings.items():
            price = self.last_price_by_ticker.get(ticker)
            if self._is_valid_price(price):
                total += float(position["shares"]) * price
        return round(float(total), 4)

    def portfolio_value(self):
        return self._portfolio_value()

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
        self.step_index += 1

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
            atr_pct = max(float(rec.get("atr_pct", 0.0)), 0.0)

            if self._is_valid_price(price):
                position["peak_price"] = max(float(position.get("peak_price", price)), price)
            position.setdefault("tp1_taken", False)

            trailing_stop_pct = self.stop_loss_pct
            if atr_pct > 0:
                trailing_stop_pct = max(self.stop_loss_pct * 0.5, min(0.35, atr_pct * self.trailing_atr_mult))
            trailing_stop = float(position.get("peak_price", price)) * (1 - trailing_stop_pct)
            hard_stop = avg_cost * (1 - self.stop_loss_pct)

            stop_loss_triggered = price <= max(hard_stop, trailing_stop)
            take_profit_triggered = price >= avg_cost * (1 + self.take_profit_pct)

            if rec.get("decision") == "SELL":
                self.sell_streak_by_ticker[ticker] = self.sell_streak_by_ticker.get(ticker, 0) + 1
            else:
                self.sell_streak_by_ticker[ticker] = 0
            explicit_sell = self.sell_streak_by_ticker.get(ticker, 0) >= self.sell_confirm_steps

            if take_profit_triggered and not position.get("tp1_taken", False):
                self._sell_fraction(
                    timestamp,
                    ticker,
                    price,
                    fraction=self.take_profit_partial_ratio,
                    action="SELL_PARTIAL_TP",
                )
                if ticker in self.holdings:
                    self.holdings[ticker]["tp1_taken"] = True

            if explicit_sell or stop_loss_triggered:
                self._sell_all(timestamp, ticker, price)

        # 2) Open new BUY positions across strongest candidates.
        buy_candidates = [
            a
            for a in analyses
            if a["decision"] == "BUY"
            and float(a.get("score", 0.0)) >= self.min_buy_score
            and self._is_valid_price(a["price"])
        ]
        buy_candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        buy_candidates = [
            a
            for a in buy_candidates
            if (self.step_index - self.last_sell_step_by_ticker.get(a["ticker"], -10_000))
            > self.cooldown_after_sell_steps
        ]

        open_slots = max(self.max_positions - len(self.holdings), 0)
        new_candidates = [c for c in buy_candidates if c["ticker"] not in self.holdings][:open_slots]

        if self.cash > 0:
            equity = self._portfolio_value()
            cap_per_position = equity * self.max_allocation_per_position
            exposure_budget_remaining = min(self.cash, equity * self.max_new_exposure_per_step)

            if new_candidates and exposure_budget_remaining > 0:
                weights = [self._buy_weight(c) for c in new_candidates]
                total_weight = sum(weights) if weights else 0.0
                planned_new_budget = exposure_budget_remaining
                for candidate, weight in zip(new_candidates, weights):
                    if exposure_budget_remaining <= 0:
                        break
                    target_budget = (
                        planned_new_budget * (weight / total_weight)
                        if total_weight > 0
                        else planned_new_budget / max(len(new_candidates), 1)
                    )
                    budget = min(target_budget, cap_per_position, self.cash, exposure_budget_remaining)
                    spent = self._buy(timestamp, ticker=candidate["ticker"], price=float(candidate["price"]), budget=budget)
                    exposure_budget_remaining -= spent

            # 3) Opportunistic averaging-up for strong signals if under-allocated.
            for candidate in buy_candidates:
                if exposure_budget_remaining <= 0:
                    break
                ticker = candidate["ticker"]
                if ticker not in self.holdings:
                    continue
                if float(candidate.get("score", 0.0)) < 1.5:
                    continue

                price = float(candidate["price"])
                avg_cost = float(self.holdings[ticker]["avg_cost"])
                if price < avg_cost * 1.01:
                    continue

                market_value = self.holdings[ticker]["shares"] * price
                room = max(cap_per_position - market_value, 0.0)
                if room <= 0:
                    continue
                budget = min(room, self.cash * 0.35, exposure_budget_remaining)
                spent = self._buy(timestamp, ticker=ticker, price=price, budget=budget)
                exposure_budget_remaining -= spent

        self.history.append(self._portfolio_value())

    def history_df(self):
        return pd.DataFrame(
            {"Step": list(range(1, len(self.history) + 1)), "Portfolio Value": self.history}
        )

    def transactions_df(self):
        return pd.DataFrame(self.transactions)
