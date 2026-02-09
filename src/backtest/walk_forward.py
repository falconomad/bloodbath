from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd

from src.analysis.dip import dip_bonus
from src.analysis.technicals import calculate_technicals
from src.api.data_fetcher import get_bulk_price_data
from src.core.sp500_list import TOP20
from src.core.top20_manager import Top20AutoManager
from src.settings import (
    SIGNAL_BUY_THRESHOLD,
    SIGNAL_SELL_THRESHOLD,
    TOP20_MIN_BUY_SCORE,
    TOP20_STARTING_CAPITAL,
    TOP20_SLIPPAGE_BPS,
    TOP20_FEE_BPS,
)


def _trend_to_score(trend: str) -> float:
    if trend == "BULLISH":
        return 1.0
    if trend == "BEARISH":
        return -1.0
    return 0.0


def _recent_growth_score(data: pd.DataFrame, lookback: int = 20) -> float:
    close = data["Close"].dropna()
    if len(close) < 2:
        return 0.0
    window = close.tail(lookback + 1)
    if len(window) < 2:
        window = close.tail(2)
    start = float(window.iloc[0])
    end = float(window.iloc[-1])
    if start <= 0:
        return 0.0
    return (end - start) / start


def _max_drawdown(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return 0.0
    running_max = series.cummax()
    dd = (series / running_max) - 1.0
    return float(dd.min())


def _daily_returns(values: Iterable[float]) -> pd.Series:
    series = pd.Series(list(values), dtype=float)
    if len(series) < 2:
        return pd.Series(dtype=float)
    return series.pct_change().dropna()


@dataclass
class BacktestResult:
    history: pd.DataFrame
    transactions: pd.DataFrame
    metrics: dict


def run_walk_forward_backtest(
    price_map: Dict[str, pd.DataFrame],
    starting_capital: float = TOP20_STARTING_CAPITAL,
    warmup_bars: int = 60,
    signal_buy_threshold: float = SIGNAL_BUY_THRESHOLD,
    signal_sell_threshold: float = SIGNAL_SELL_THRESHOLD,
    min_buy_score: float = TOP20_MIN_BUY_SCORE,
    slippage_bps: float = TOP20_SLIPPAGE_BPS,
    fee_bps: float = TOP20_FEE_BPS,
) -> BacktestResult:
    manager = Top20AutoManager(
        starting_capital=starting_capital,
        min_buy_score=min_buy_score,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
    )

    valid_map: Dict[str, pd.DataFrame] = {}
    for ticker, df in price_map.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        clean = df.copy()
        clean.index = pd.to_datetime(clean.index, errors="coerce")
        clean = clean[clean.index.notna()].sort_index()
        if clean.empty:
            continue
        valid_map[ticker] = clean

    if not valid_map:
        metrics = {
            "starting_capital": float(starting_capital),
            "ending_value": float(starting_capital),
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility_pct": 0.0,
            "sharpe_like": 0.0,
            "win_rate_daily_pct": 0.0,
            "turnover_ratio": 0.0,
            "num_cycles": 0,
            "num_trades": 0,
        }
        return BacktestResult(history=pd.DataFrame(columns=["date", "value"]), transactions=pd.DataFrame(), metrics=metrics)

    all_dates = sorted(set().union(*[set(df.index) for df in valid_map.values()]))
    cycle_rows = []

    for i, current_date in enumerate(all_dates):
        if i < warmup_bars:
            continue

        analyses = []
        for ticker, df in valid_map.items():
            window = df[df.index <= current_date]
            if len(window) < warmup_bars:
                continue
            close = window["Close"].dropna()
            if close.empty:
                continue
            price = float(close.iloc[-1])

            trend = calculate_technicals(window)
            trend_score = _trend_to_score(trend)
            dip_score, _drawdown, _stabilized, vol_penalty = dip_bonus(window)
            composite_score = trend_score
            final_score = composite_score + dip_score + vol_penalty

            if final_score >= signal_buy_threshold:
                decision = "BUY"
            elif final_score <= signal_sell_threshold:
                decision = "SELL"
            else:
                decision = "HOLD"

            analyses.append(
                {
                    "ticker": ticker,
                    "decision": decision,
                    "score": round(final_score, 4),
                    "price": price,
                    "sentiment": 0.0,
                    "growth_20d": round(_recent_growth_score(window, 20), 4),
                }
            )

        manager.step(analyses)
        cycle_rows.append({"date": current_date, "value": manager.portfolio_value()})

    history = pd.DataFrame(cycle_rows)
    transactions = manager.transactions_df()

    values = history["value"].tolist() if not history.empty else [float(starting_capital)]

    benchmark_values = []
    if not history.empty:
        start_date = history["date"].iloc[0]
        bases = {}
        for ticker, df in valid_map.items():
            base_window = df[df.index <= start_date]["Close"].dropna()
            if not base_window.empty and float(base_window.iloc[-1]) > 0:
                bases[ticker] = float(base_window.iloc[-1])

        for d in history["date"]:
            rels = []
            for ticker, base in bases.items():
                w = valid_map[ticker][valid_map[ticker].index <= d]["Close"].dropna()
                if w.empty:
                    continue
                rels.append(float(w.iloc[-1]) / base)
            benchmark_values.append(float(starting_capital) * (sum(rels) / len(rels)) if rels else float(starting_capital))
    ending_value = float(values[-1]) if values else float(starting_capital)
    total_return = (ending_value / float(starting_capital) - 1.0) if starting_capital > 0 else 0.0
    daily_rets = _daily_returns(values)
    annualized_vol = float(daily_rets.std() * (252 ** 0.5)) if not daily_rets.empty else 0.0
    sharpe_like = float((daily_rets.mean() / daily_rets.std()) * (252 ** 0.5)) if len(daily_rets) > 1 and daily_rets.std() > 0 else 0.0
    win_rate_daily = float((daily_rets > 0).mean()) if not daily_rets.empty else 0.0
    max_dd = _max_drawdown(values)

    num_days = len(values)
    years = num_days / 252 if num_days > 0 else 0.0
    cagr = ((ending_value / float(starting_capital)) ** (1 / years) - 1.0) if years > 0 and starting_capital > 0 else 0.0

    traded_value = 0.0
    if not transactions.empty:
        traded_value = float((transactions["shares"] * transactions["price"]).abs().sum())
    avg_equity = float(pd.Series(values).mean()) if values else float(starting_capital)
    turnover = traded_value / avg_equity if avg_equity > 0 else 0.0

    benchmark_ending = benchmark_values[-1] if benchmark_values else float(starting_capital)
    benchmark_return = (benchmark_ending / float(starting_capital) - 1.0) if starting_capital > 0 else 0.0
    excess_return = total_return - benchmark_return

    metrics = {
        "starting_capital": round(float(starting_capital), 4),
        "ending_value": round(ending_value, 4),
        "total_return_pct": round(total_return * 100, 2),
        "benchmark_return_pct": round(benchmark_return * 100, 2),
        "excess_return_pct": round(excess_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "volatility_pct": round(annualized_vol * 100, 2),
        "sharpe_like": round(sharpe_like, 4),
        "win_rate_daily_pct": round(win_rate_daily * 100, 2),
        "turnover_ratio": round(float(turnover), 4),
        "num_cycles": int(len(history)),
        "num_trades": int(len(transactions)),
    }

    return BacktestResult(history=history, transactions=transactions, metrics=metrics)


def run_backtest_for_top20(period: str = "2y", interval: str = "1d") -> BacktestResult:
    price_map = get_bulk_price_data(TOP20, period=period, interval=interval)
    return run_walk_forward_backtest(price_map, starting_capital=TOP20_STARTING_CAPITAL)
