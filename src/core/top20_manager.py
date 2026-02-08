
import datetime
import pandas as pd
from src.core.sp500_list import TOP20

class Top20AutoManager:
    def __init__(self, starting_capital=500):
        self.cash = starting_capital
        self.position = None
        self.shares = 0
        self.history = []
        self.transactions = []

    def choose_best(self, analyses):
        return max(analyses, key=lambda x: x["score"])

    def step(self, analyses):
        best = self.choose_best(analyses)
        ticker = best["ticker"]
        decision = best["decision"]
        price = best["price"]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if decision == "BUY" and self.cash > price:
            shares = int(self.cash // price)
            self.cash -= shares * price
            self.shares = shares
            self.position = ticker

            self.transactions.append({
                "time": timestamp,
                "ticker": ticker,
                "action": "BUY",
                "shares": shares,
                "price": round(price,2)
            })

        elif decision == "SELL" and self.position:
            self.cash += self.shares * price
            self.transactions.append({
                "time": timestamp,
                "ticker": self.position,
                "action": "SELL",
                "shares": self.shares,
                "price": round(price,2)
            })
            self.position = None
            self.shares = 0

        portfolio_value = self.cash
        if self.position:
            portfolio_value += self.shares * price

        self.history.append(portfolio_value)

    def history_df(self):
        return pd.DataFrame({
            "Step": list(range(1, len(self.history)+1)),
            "Portfolio Value": self.history
        })

    def transactions_df(self):
        return pd.DataFrame(self.transactions)
