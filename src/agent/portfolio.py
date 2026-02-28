from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortfolioSnapshot:
    value: float
    cash: float
    holdings_count: int


def build_portfolio_snapshot(portfolio_value: float, cash: float, holdings_count: int) -> PortfolioSnapshot:
    return PortfolioSnapshot(value=float(portfolio_value), cash=float(cash), holdings_count=int(holdings_count))
