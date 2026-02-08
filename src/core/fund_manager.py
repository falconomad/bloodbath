
import pandas as pd
import json
import os

class AutoFundManager:
    def __init__(self, starting_capital=500):
        self.cash = starting_capital
        self.history = []
        self.day = 0

    def step(self, decision, price):
        # Simple autonomous logic: invest 10% of cash on BUY, exit on SELL
        if decision == "BUY" and self.cash > 0:
            investment = self.cash * 0.1
            self.cash -= investment

        if decision == "SELL":
            self.cash += self.cash * 0.05  # assume small profit taking

        self.day += 1
        self.history.append(self.cash)

    def get_history(self):
        return pd.DataFrame({
            "Day": list(range(1, len(self.history)+1)),
            "Portfolio Value": self.history
        })
