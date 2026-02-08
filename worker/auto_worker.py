from pathlib import Path
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

# Ensure repo root is on sys.path when running as `python worker/auto_worker.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db import claim_worker_run, get_connection, init_db
from src.advisor import run_top20_cycle

DB_AVAILABLE = True

# Initialize DB (safe to call multiple times)
try:
    init_db()
except Exception as exc:
    DB_AVAILABLE = False
    print(f"Database initialization failed; continuing without persistence: {exc}")


def save(history, transactions, positions):
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

    if not positions.empty:
        snapshot_time = str(positions.iloc[0]["time"])
        c.execute("DELETE FROM position_snapshots WHERE time = %s", (snapshot_time,))

        for _, p in positions.iterrows():
            c.execute(
                """
                INSERT INTO position_snapshots (
                    time, ticker, shares, avg_cost, current_price,
                    market_value, allocation, pnl, pnl_pct
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    p["time"],
                    p["ticker"],
                    int(p["shares"]),
                    float(p["avg_cost"]),
                    float(p["current_price"]),
                    float(p["market_value"]),
                    float(p["allocation"]),
                    float(p["pnl"]),
                    float(p["pnl_pct"]),
                ),
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


def build_run_key(now=None):
    """Create a stable run key to avoid duplicate trades from cron retries.

    We use 10-minute buckets because the workflow is typically scheduled every 10 minutes.
    """
    current = now or datetime.now(ZoneInfo("Europe/Paris"))
    bucket_minute = (current.minute // 10) * 10
    return f"{current.strftime('%Y-%m-%d')}-{current.hour:02d}-{bucket_minute:02d}"


def main():
    if not is_market_active_cet():
        print("Outside market window; skipping analysis and actions.")
        return

    if DB_AVAILABLE:
        run_key = build_run_key()
        if not claim_worker_run(run_key):
            print(f"Run key {run_key} already processed; skipping duplicate execution.")
            return

    history, transactions, positions = run_top20_cycle()
    save(history, transactions, positions)
    print("Cycle complete.")


if __name__ == "__main__":
    main()
