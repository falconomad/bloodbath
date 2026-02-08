
import time
import sqlite3
from src.advisor import run_top20_cycle

DB = "db/portfolio.db"

def save(history, transactions):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    if not history.empty:
        last = history.iloc[-1]
        c.execute("INSERT INTO portfolio VALUES (?, ?)", (str(last['Step']), float(last['Portfolio Value'])))

    if not transactions.empty:
        t = transactions.iloc[-1]
        c.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?)",
                  (t['time'], t['ticker'], t['action'], int(t['shares']), float(t['price'])))

    conn.commit()
    conn.close()

print("Autonomous Worker Running...")

while True:
    history, transactions = run_top20_cycle()
    save(history, transactions)

    print("Cycle complete. Sleeping 1 hour...")
    time.sleep(3600)
