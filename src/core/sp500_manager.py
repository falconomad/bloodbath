
import pandas as pd
import datetime

class SP500AutoManager:
    def __init__(self, starting_capital=500):
        self.cash = starting_capital
        self.shares = 0
        self.history = []
        self.transactions = []

    def step(self, decision, price):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if decision == "BUY" and self.cash > price:
            shares_to_buy = int(self.cash // price)
            cost = shares_to_buy * price
            self.cash -= cost
            self.shares += shares_to_buy

            self.transactions.append({
                "time": timestamp,
                "action": "BUY",
                "shares": shares_to_buy,
                "price": round(price,2),
                "cash_after": round(self.cash,2)
            })

        elif decision == "SELL" and self.shares > 0:
            revenue = self.shares * price
            sold = self.shares
            self.cash += revenue
            self.shares = 0

            self.transactions.append({
                "time": timestamp,
                "action": "SELL",
                "shares": sold,
                "price": round(price,2),
                "cash_after": round(self.cash,2)
            })

        portfolio_value = self.cash + (self.shares * price)
        self.history.append(portfolio_value)

    def get_history_df(self):
        return pd.DataFrame({
            "Step": list(range(1, len(self.history)+1)),
            "Portfolio Value": self.history
        })

    def get_transactions_df(self):
        return pd.DataFrame(self.transactions)
