from pathlib import Path
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

# Ensure repo root is on sys.path when running as `python worker/auto_worker.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db import get_connection, init_db
from src.advisor import run_top20_cycle

DB_AVAILABLE = True

# Initialize DB (safe to call multiple times)
try:
    init_db()
except Exception as exc:
    DB_AVAILABLE = False
    print(f"Database initialization failed; continuing without persistence: {exc}")


def save(history, transactions):
    if not DB_AVAILABLE:
        print("Skipping DB save because database is unavailable.")
        return

    conn = get_connection()
    c = conn.cursor()

    if not history.empty:
        last = history.iloc[-1]
        c.execute(
            "INSERT INTO portfolio (time, value) VALUES (%s, %s)",
            (str(last["Step"]), float(last["Portfolio Value"])),
        )

    if not transactions.empty:
        for _, t in transactions.iterrows():
            c.execute(
                "INSERT INTO transactions (time, ticker, action, shares, price) VALUES (%s, %s, %s, %s, %s)",
                (t["time"], t["ticker"], t["action"], int(t["shares"]), float(t["price"])),
            )

    conn.commit()
    c.close()
    conn.close()


print("GitHub Actions worker starting one execution cycle...")


def is_market_active_cet(now=None):
    """Return True when the market window is open in CET/CEST.

    Defaults to Monday-Friday between 15:30 and 22:00 in Europe/Paris
    (15:30-22:00 CET in winter, 15:30-22:00 CEST in summer).
    """
    current = now or datetime.now(ZoneInfo("Europe/Paris"))

    # Monday=0, Sunday=6
    if current.weekday() >= 5:
        return False

    market_open = time(15, 30)
    market_close = time(22, 0)
    now_time = current.time().replace(tzinfo=None)

    return market_open <= now_time <= market_close


def is_opening_window_cet(now=None):
    """Run once shortly after market open to avoid repeated intra-day re-entry."""
    current = now or datetime.now(ZoneInfo("Europe/Paris"))
    if current.weekday() >= 5:
        return False

    now_time = current.time().replace(tzinfo=None)
    return time(15, 30) <= now_time < time(15, 40)


def main():
    if is_market_active_cet() and is_opening_window_cet():
        history, transactions = run_top20_cycle()
        save(history, transactions)
        print("Cycle complete.")
    else:
        print("Outside opening execution window; skipping analysis and actions.")


if __name__ == "__main__":
    main()
