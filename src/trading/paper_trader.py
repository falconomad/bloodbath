
import pandas as pd

class VirtualPortfolio:
    def __init__(self, starting_capital=500):
        self.cash = starting_capital
        self.holdings = {}
        self.trade_log = []
        self.balance_history = []

    def current_value(self, market_prices):
        value = self.cash
        for ticker, shares in self.holdings.items():
            if ticker in market_prices:
                value += shares * market_prices[ticker]
        return value

    def buy(self, ticker, price):
        if self.cash >= price:
            self.cash -= price
            self.holdings[ticker] = self.holdings.get(ticker, 0) + 1
            self.trade_log.append(f"BUY {ticker} at {price}")
            return True
        return False

    def sell(self, ticker, price):
        if ticker in self.holdings and self.holdings[ticker] > 0:
            self.holdings[ticker] -= 1
            self.cash += price
            self.trade_log.append(f"SELL {ticker} at {price}")
            return True
        return False

    def record_day(self, market_prices):
        total = self.current_value(market_prices)
        self.balance_history.append(total)
        return total
