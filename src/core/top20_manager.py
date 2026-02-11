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
        buy_confirm_steps=1,
        max_entry_growth_20d=0.18,
        initial_entry_ratio=1.0,
        max_sector_allocation=0.45,
        max_buy_exposure_per_sector_step=0.30,
        max_positions_per_sector=2,
        enable_position_rotation=False,
        rotation_min_score_gap=0.15,
        rotation_sell_fraction=0.35,
        rotation_max_swaps_per_step=1,
        max_daily_loss_pct=0.05,
        min_risk_reward_ratio=1.5,
        rebalance_tolerance=0.02,
        max_portfolio_correlation=0.85,
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
        self.buy_confirm_steps = int(buy_confirm_steps)
        self.max_entry_growth_20d = float(max_entry_growth_20d)
        self.initial_entry_ratio = float(initial_entry_ratio)
        self.max_sector_allocation = float(max_sector_allocation)
        self.max_buy_exposure_per_sector_step = float(max_buy_exposure_per_sector_step)
        self.max_positions_per_sector = int(max_positions_per_sector)
        self.enable_position_rotation = bool(enable_position_rotation)
        self.rotation_min_score_gap = float(rotation_min_score_gap)
        self.rotation_sell_fraction = float(rotation_sell_fraction)
        self.rotation_max_swaps_per_step = int(rotation_max_swaps_per_step)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.min_risk_reward_ratio = float(min_risk_reward_ratio)
        self.rebalance_tolerance = float(rebalance_tolerance)
        self.max_portfolio_correlation = float(max_portfolio_correlation)
        # ticker -> {"shares": float, "avg_cost": float, "peak_price": float}
        self.holdings = {}
        self.last_price_by_ticker = {}
        self.last_sell_step_by_ticker = {}
        self.sell_streak_by_ticker = {}
        self.buy_streak_by_ticker = {}
        self.step_index = 0
        self.history = []
        self.transactions = []
        self._current_day = None
        self._day_start_equity = None
        self._daily_loss_triggered = False

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
        atr_pct = max(float(candidate.get("atr_pct", 0.0)), 0.0)

        # Blend conviction (score), sentiment, and recent growth into a positive weight.
        sentiment_factor = max(0.6, min(1.4, 1.0 + (0.30 * sentiment)))
        growth_factor = max(0.7, min(1.5, 1.0 + (0.80 * growth)))
        # Vol-adjusted sizing: higher volatility gets a smaller size.
        volatility_factor = 1.0 / (1.0 + (4.0 * atr_pct))
        volatility_factor = max(0.35, min(1.15, volatility_factor))
        return max(score * sentiment_factor * growth_factor * volatility_factor, 1e-6)

    def _stop_pct_for(self, atr_pct):
        atr_pct = max(float(atr_pct), 0.0)
        if atr_pct > 0:
            return max(self.stop_loss_pct * 0.5, min(0.35, atr_pct * self.trailing_atr_mult))
        return self.stop_loss_pct

    def _risk_reward_ratio(self, candidate):
        stop_pct = self._stop_pct_for(float(candidate.get("atr_pct", 0.0)))
        expected_upside = float(
            candidate.get("expected_return_10d", candidate.get("expected_return", self.take_profit_pct))
        )
        if expected_upside <= 0:
            return 0.0
        return expected_upside / max(stop_pct, 1e-6)

    def _rebalance_positions(self, timestamp, analysis_by_ticker):
        if not self.holdings:
            return
        equity = self._portfolio_value()
        if equity <= 0:
            return
        max_allocation = self.max_allocation_per_position + self.rebalance_tolerance
        for ticker in list(self.holdings.keys()):
            price = float(self.last_price_by_ticker.get(ticker, self.holdings[ticker].get("avg_cost", 0.0)))
            if not self._is_valid_price(price):
                continue
            market_value = float(self.holdings[ticker]["shares"]) * price
            allocation = market_value / equity
            if allocation <= max_allocation:
                continue
            target_value = equity * self.max_allocation_per_position
            excess_value = max(market_value - target_value, 0.0)
            if excess_value <= 0:
                continue
            fraction_to_trim = max(min(excess_value / market_value, 0.95), 0.0)
            if fraction_to_trim > 0:
                self._sell_fraction(timestamp, ticker, price, fraction_to_trim, action="SELL_REBALANCE")

    def _buy(self, timestamp, ticker, price, budget, sector=None):
        if not self._is_valid_price(price) or budget <= 0 or self.cash <= 0:
            return 0.0

        exec_price = float(price) * (1.0 + (self.slippage_bps / 10_000.0))
        fee_rate = self.fee_bps / 10_000.0
        # Ensure budget check includes fees so cash cannot go negative from execution costs.
        shares = min(budget, self.cash) / (exec_price * (1.0 + fee_rate))
        if shares <= 0 or shares < 1e-6:
            return 0.0

        notional = shares * exec_price
        fee = notional * fee_rate
        total_cost = notional + fee
        self.cash -= total_cost
        if self.cash < 0 and abs(self.cash) < 1e-6:
            self.cash = 0.0

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
            "sector": str(sector or position.get("sector", "UNKNOWN")),
        }
        self._record(timestamp, ticker, "BUY", shares, exec_price)
        return total_cost

    def _sector(self, ticker, analysis_by_ticker):
        if ticker in self.holdings:
            existing = str(self.holdings[ticker].get("sector", "")).strip()
            if existing:
                return existing
        rec = analysis_by_ticker.get(ticker, {})
        return str(rec.get("sector", "UNKNOWN")).strip() or "UNKNOWN"

    def _sector_market_value(self, sector_name, analysis_by_ticker):
        total = 0.0
        for ticker, position in self.holdings.items():
            if self._sector(ticker, analysis_by_ticker) != sector_name:
                continue
            price = float(self.last_price_by_ticker.get(ticker, position.get("avg_cost", 0.0)))
            if not self._is_valid_price(price):
                continue
            total += float(position["shares"]) * price
        return total

    def _sector_position_count(self, sector_name, analysis_by_ticker):
        return sum(1 for ticker in self.holdings if self._sector(ticker, analysis_by_ticker) == sector_name)

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
        total_positions_value = 0.0
        for ticker, position in self.holdings.items():
            shares = float(position["shares"])
            avg_cost = float(position["avg_cost"])
            current_price = float(self.last_price_by_ticker.get(ticker, avg_cost))
            total_positions_value += shares * current_price
        rows = []
        for ticker, position in self.holdings.items():
            shares = float(position["shares"])
            avg_cost = float(position["avg_cost"])
            current_price = float(self.last_price_by_ticker.get(ticker, avg_cost))
            market_value = shares * current_price
            cost_basis = shares * avg_cost
            pnl = market_value - cost_basis
            allocation = (market_value / total_positions_value) if total_positions_value > 0 else 0.0
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
        current_day = timestamp[:10]
        if current_day != self._current_day:
            self._current_day = current_day
            self._day_start_equity = self._portfolio_value()
            self._daily_loss_triggered = False

        for analysis in analyses:
            price = float(analysis["price"])
            if self._is_valid_price(price):
                self.last_price_by_ticker[analysis["ticker"]] = price

        analysis_by_ticker = {a["ticker"]: a for a in analyses}
        equity_before_actions = self._portfolio_value()
        if self._day_start_equity is None:
            self._day_start_equity = equity_before_actions
        if self._day_start_equity > 0:
            daily_drawdown = (self._day_start_equity - equity_before_actions) / self._day_start_equity
            if daily_drawdown >= self.max_daily_loss_pct:
                self._daily_loss_triggered = True

        if self._daily_loss_triggered:
            for ticker in list(self.holdings.keys()):
                price = float(self.last_price_by_ticker.get(ticker, self.holdings[ticker].get("avg_cost", 0.0)))
                if self._is_valid_price(price):
                    self._sell_all(timestamp, ticker, price)
            self.history.append(self._portfolio_value())
            return

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

            trailing_stop_pct = self._stop_pct_for(atr_pct)
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
        buy_candidates = []
        for a in analyses:
            ticker = a["ticker"]
            if a["decision"] == "BUY":
                self.buy_streak_by_ticker[ticker] = self.buy_streak_by_ticker.get(ticker, 0) + 1
            else:
                self.buy_streak_by_ticker[ticker] = 0

            if a["decision"] != "BUY":
                continue
            if float(a.get("score", 0.0)) < self.min_buy_score:
                continue
            if not self._is_valid_price(a["price"]):
                continue
            if self.buy_streak_by_ticker.get(ticker, 0) < self.buy_confirm_steps:
                continue

            growth_20d = float(a.get("growth_20d", 0.0))
            if growth_20d > self.max_entry_growth_20d:
                continue
            if self._risk_reward_ratio(a) < self.min_risk_reward_ratio:
                continue
            corr_to_portfolio = abs(float(a.get("corr_to_portfolio", 0.0)))
            if corr_to_portfolio > self.max_portfolio_correlation:
                continue

            buy_candidates.append(a)
        buy_candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        buy_candidates = [
            a
            for a in buy_candidates
            if (self.step_index - self.last_sell_step_by_ticker.get(a["ticker"], -10_000))
            > self.cooldown_after_sell_steps
        ]

        open_slots = max(self.max_positions - len(self.holdings), 0)
        new_candidates = [c for c in buy_candidates if c["ticker"] not in self.holdings][:open_slots]
        new_candidates = [
            c
            for c in new_candidates
            if self._sector_position_count(self._sector(c["ticker"], analysis_by_ticker), analysis_by_ticker)
            < self.max_positions_per_sector
        ]

        # Optional: rotate out weaker holdings to free capital for stronger new BUY candidates.
        if self.enable_position_rotation and self.cash <= 1e-6 and buy_candidates:
            swaps = 0
            held_rank = []
            for held_ticker in list(self.holdings.keys()):
                held_rec = analysis_by_ticker.get(held_ticker, {})
                held_score = float(held_rec.get("score", 0.0))
                held_rank.append((held_ticker, held_score))
            held_rank.sort(key=lambda x: x[1])  # weakest first
            outside_candidates = [c for c in buy_candidates if c["ticker"] not in self.holdings]
            outside_candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)

            while swaps < max(self.rotation_max_swaps_per_step, 0) and held_rank and outside_candidates:
                weakest_ticker, weakest_score = held_rank[0]
                strongest = outside_candidates[0]
                strongest_score = float(strongest.get("score", 0.0))
                if strongest_score < (weakest_score + self.rotation_min_score_gap):
                    break
                price = float(self.last_price_by_ticker.get(weakest_ticker, self.holdings[weakest_ticker]["avg_cost"]))
                if not self._is_valid_price(price):
                    held_rank.pop(0)
                    continue
                self._sell_fraction(
                    timestamp,
                    weakest_ticker,
                    price,
                    fraction=max(min(self.rotation_sell_fraction, 0.95), 0.05),
                    action="SELL_ROTATE",
                )
                swaps += 1
                held_rank.pop(0)
                outside_candidates.pop(0)

        if self.cash > 0:
            equity = self._portfolio_value()
            cap_per_position = equity * self.max_allocation_per_position
            exposure_budget_remaining = min(self.cash, equity * self.max_new_exposure_per_step)

            if new_candidates and exposure_budget_remaining > 0:
                weights = [self._buy_weight(c) for c in new_candidates]
                total_weight = sum(weights) if weights else 0.0
                planned_new_budget = exposure_budget_remaining * self.initial_entry_ratio
                for candidate, weight in zip(new_candidates, weights):
                    if exposure_budget_remaining <= 0:
                        break
                    sector = self._sector(candidate["ticker"], analysis_by_ticker)
                    current_sector_positions = self._sector_position_count(sector, analysis_by_ticker)
                    if current_sector_positions >= self.max_positions_per_sector:
                        continue
                    sector_current_value = self._sector_market_value(sector, analysis_by_ticker)
                    sector_cap_equity = equity * self.max_sector_allocation
                    sector_cap_step = equity * self.max_buy_exposure_per_sector_step
                    sector_room = max(min(sector_cap_equity - sector_current_value, sector_cap_step), 0.0)
                    if sector_room <= 0:
                        continue
                    target_budget = (
                        planned_new_budget * (weight / total_weight)
                        if total_weight > 0
                        else planned_new_budget / max(len(new_candidates), 1)
                    )
                    budget = min(target_budget, cap_per_position, self.cash, exposure_budget_remaining, sector_room)
                    spent = self._buy(
                        timestamp,
                        ticker=candidate["ticker"],
                        price=float(candidate["price"]),
                        budget=budget,
                        sector=sector,
                    )
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
                sector = self._sector(ticker, analysis_by_ticker)
                sector_current_value = self._sector_market_value(sector, analysis_by_ticker)
                sector_cap_equity = equity * self.max_sector_allocation
                sector_room = max(sector_cap_equity - sector_current_value, 0.0)
                if sector_room <= 0:
                    continue
                budget = min(room, self.cash * 0.35, exposure_budget_remaining)
                budget = min(budget, sector_room)
                spent = self._buy(timestamp, ticker=ticker, price=price, budget=budget, sector=sector)
                exposure_budget_remaining -= spent

        self._rebalance_positions(timestamp, analysis_by_ticker)
        self.history.append(self._portfolio_value())

    def history_df(self):
        return pd.DataFrame(
            {"Step": list(range(1, len(self.history) + 1)), "Portfolio Value": self.history}
        )

    def transactions_df(self):
        return pd.DataFrame(self.transactions)
