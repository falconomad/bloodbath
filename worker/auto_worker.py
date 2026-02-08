
from src.db import get_connection, init_db
from src.advisor import run_top20_cycle

# Initialize DB (safe to call multiple times)
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


print("GitHub Actions worker starting one execution cycle...")

history, transactions = run_top20_cycle()
save(history, transactions)

print("Cycle complete.")
