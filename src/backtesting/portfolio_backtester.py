from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.backtesting.simple_backtester import (
    BacktestExample,
    ExecutionModel,
    _cost_rate,
    _max_drawdown_from_pnl,
    _years_span,
)
from src.pipeline.decision_engine import (
    aggregate_confidence,
    decide,
    resolve_effective_weights,
    weighted_score,
)


@dataclass
class PortfolioBacktestConfig:
    starting_capital: float = 1.0
    max_positions: int = 8
    max_position_size: float = 0.2


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    positive = {k: max(float(v), 0.0) for k, v in (weights or {}).items()}
    total = sum(positive.values())
    if total <= 0:
        n = max(len(positive), 1)
        return {k: 1.0 / n for k in positive} if positive else {}
    return {k: v / total for k, v in positive.items()}


def evaluate_candidate_portfolio(
    examples: list[BacktestExample],
    cfg: dict[str, Any],
    weights: dict[str, float],
    execution_model: ExecutionModel | None = None,
    portfolio_cfg: PortfolioBacktestConfig | None = None,
) -> dict[str, float]:
    if not examples:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "profit_factor": 0.0,
            "cagr": 0.0,
            "calmar_ratio": 0.0,
            "expectancy_per_trade": 0.0,
        }

    model = execution_model or ExecutionModel()
    p_cfg = portfolio_cfg or PortfolioBacktestConfig()
    local_cfg = dict(cfg or {})
    local_cfg["weights"] = _normalize_weights(weights)
    state: dict[str, dict[str, Any]] = {}
    costs = _cost_rate(model)

    cash = float(p_cfg.starting_capital)
    positions: dict[str, float] = {}
    trade_pnl: list[float] = []
    step_pnl: list[float] = []
    wins = 0
    trades = 0

    ordered = sorted(examples, key=lambda x: (x.ts, x.ticker))
    for i, ex in enumerate(ordered, start=1):
        effective_weights, _ = resolve_effective_weights(local_cfg, ex.risk_context)
        score = weighted_score(ex.signals, effective_weights)
        conf, _ = aggregate_confidence(
            ex.signals,
            local_cfg.get("risk", {}).get("conflict_penalty", 0.2),
            weights=effective_weights,
            max_conflict_drop=local_cfg.get("risk", {}).get("max_confidence_drop_on_conflict", 0.75),
        )
        decision, _reasons, size = decide(
            ticker=ex.ticker,
            score=score,
            confidence=conf,
            signals=ex.signals,
            risk_context=ex.risk_context,
            state=state,
            cycle_idx=i,
            cfg=local_cfg,
        )

        equity = cash + sum(positions.values())
        position_cap = max(float(p_cfg.max_position_size), 0.0) * max(equity, 0.0)
        target_value = min(position_cap, max(size, 0.0) * max(equity, 0.0))
        if len(positions) >= max(int(p_cfg.max_positions), 1) and ex.ticker not in positions and decision == "BUY":
            decision = "HOLD"

        ticker_pos = float(positions.get(ex.ticker, 0.0))
        trade_ret = 0.0
        if decision == "BUY" and target_value > ticker_pos and cash > 0:
            add_value = min(target_value - ticker_pos, cash)
            if add_value > 0:
                executed = add_value * max(min(float(model.fill_ratio), 1.0), 0.0)
                fee = executed * costs
                cash -= (executed + fee)
                positions[ex.ticker] = ticker_pos + executed
                pnl = executed * float(ex.forward_return) - fee
                positions[ex.ticker] += pnl
                trade_ret = pnl / max(equity, 1e-9)
        elif decision == "SELL" and ticker_pos > 0:
            executed = ticker_pos * max(min(float(model.fill_ratio), 1.0), 0.0)
            fee = executed * costs
            pnl = (-executed * float(ex.forward_return)) - fee
            cash += (executed - fee + pnl)
            positions.pop(ex.ticker, None)
            trade_ret = pnl / max(equity, 1e-9)
        else:
            if ticker_pos > 0:
                pnl = ticker_pos * float(ex.forward_return)
                positions[ex.ticker] = ticker_pos + pnl
                trade_ret = pnl / max(equity, 1e-9)

        step_pnl.append(trade_ret)
        if decision in {"BUY", "SELL"}:
            trades += 1
            trade_pnl.append(trade_ret)
            if trade_ret > 0:
                wins += 1

    total_return = (cash + sum(positions.values()) - p_cfg.starting_capital) / max(p_cfg.starting_capital, 1e-9)
    avg = sum(step_pnl) / len(step_pnl)
    variance = sum((x - avg) ** 2 for x in step_pnl) / len(step_pnl)
    std = math.sqrt(max(variance, 0.0))
    downside = [x for x in step_pnl if x < 0]
    downside_std = math.sqrt(sum(x * x for x in downside) / len(downside)) if downside else 0.0
    sharpe = (avg / std) if std > 1e-9 else 0.0
    sortino = (avg / downside_std) if downside_std > 1e-9 else 0.0
    max_drawdown = _max_drawdown_from_pnl(step_pnl)
    wins_list = [x for x in trade_pnl if x > 0]
    losses_list = [x for x in trade_pnl if x < 0]
    gross_profit = sum(wins_list)
    gross_loss_abs = abs(sum(losses_list))
    win_rate = (wins / trades) if trades > 0 else 0.0
    avg_win = (sum(wins_list) / len(wins_list)) if wins_list else 0.0
    avg_loss = (sum(losses_list) / len(losses_list)) if losses_list else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 1e-12 else (999.0 if gross_profit > 0 else 0.0)
    years = _years_span(examples)
    equity_final = 1.0 + total_return
    cagr = ((equity_final ** (1.0 / years)) - 1.0) if years > 1e-9 and equity_final > 0 else 0.0
    calmar = (cagr / max_drawdown) if max_drawdown > 1e-9 else 0.0
    return {
        "trades": float(trades),
        "win_rate": float(win_rate),
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "profit_factor": float(profit_factor),
        "cagr": float(cagr),
        "calmar_ratio": float(calmar),
        "expectancy_per_trade": float(expectancy),
    }
