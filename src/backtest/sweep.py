from __future__ import annotations

from itertools import product
from typing import Dict, List

import pandas as pd

from src.backtest.walk_forward import run_walk_forward_backtest


def run_parameter_sweep(
    price_map: Dict[str, pd.DataFrame],
    *,
    buy_thresholds: List[float],
    sell_thresholds: List[float],
    min_buy_scores: List[float],
    slippage_bps_values: List[float],
    fee_bps_values: List[float],
    max_drawdown_limit_pct: float = -35.0,
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []

    for buy_th, sell_th, min_buy, slip_bps, fee_bps in product(
        buy_thresholds, sell_thresholds, min_buy_scores, slippage_bps_values, fee_bps_values
    ):
        if sell_th >= 0 or buy_th <= 0:
            continue
        if abs(sell_th) > buy_th * 2.5:
            continue

        result = run_walk_forward_backtest(
            price_map,
            signal_buy_threshold=buy_th,
            signal_sell_threshold=sell_th,
            min_buy_score=min_buy,
            slippage_bps=slip_bps,
            fee_bps=fee_bps,
        )
        m = result.metrics
        if m["max_drawdown_pct"] < max_drawdown_limit_pct:
            continue

        rows.append(
            {
                "buy_threshold": buy_th,
                "sell_threshold": sell_th,
                "min_buy_score": min_buy,
                "slippage_bps": slip_bps,
                "fee_bps": fee_bps,
                "ending_value": m["ending_value"],
                "total_return_pct": m["total_return_pct"],
                "benchmark_return_pct": m["benchmark_return_pct"],
                "excess_return_pct": m["excess_return_pct"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "volatility_pct": m["volatility_pct"],
                "sharpe_like": m["sharpe_like"],
                "turnover_ratio": m["turnover_ratio"],
                "num_trades": m["num_trades"],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "buy_threshold",
                "sell_threshold",
                "min_buy_score",
                "slippage_bps",
                "fee_bps",
                "ending_value",
                "total_return_pct",
                "benchmark_return_pct",
                "excess_return_pct",
                "max_drawdown_pct",
                "volatility_pct",
                "sharpe_like",
                "turnover_ratio",
                "num_trades",
            ]
        )

    out = pd.DataFrame(rows).sort_values(
        by=["excess_return_pct", "sharpe_like", "max_drawdown_pct"],
        ascending=[False, False, False],
    )
    return out.head(top_k).reset_index(drop=True)
