from src.db import get_connection, init_db
import time
from src.advisor import run_top20_cycle

# Ensure tables exist in PostgreSQL
init_db()

def save(history, transactions):
    conn = get_connection()
    c = conn.cursor()

    if not history.empty:
        last = history.iloc[-1]
        c.execute(
            "INSERT INTO portfolio (time, value) VALUES (%s, %s)",
            (str(last['Step']), float(last['Portfolio Value']))
        )

    if not transactions.empty:
        t = transactions.iloc[-1]
        c.execute(
            "INSERT INTO transactions (time, ticker, action, shares, price) VALUES (%s, %s, %s, %s, %s)",
            (t['time'], t['ticker'], t['action'], int(t['shares']), float(t['price']))
        )

    conn.commit()
    c.close()
    conn.close()


print("Autonomous Worker Running (PostgreSQL Mode)...")

while True:
    history, transactions = run_top20_cycle()
    save(history, transactions)

    print("Cycle complete. Sleeping 1 hour...")
    time.sleep(3600)
