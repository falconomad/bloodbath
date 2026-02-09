from pathlib import Path
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

# Ensure repo root is on sys.path when running as `python worker/auto_worker.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db import claim_worker_run, get_connection, init_db
from src.advisor import run_top20_cycle_with_signals, top20_manager
from src.settings import TRADE_MODE

DB_AVAILABLE = True

# Initialize DB (safe to call multiple times)
try:
    init_db()
except Exception as exc:
    DB_AVAILABLE = False
    print(f"Database initialization failed; continuing without persistence: {exc}")


def save(history, transactions, positions, analyses):
    if not DB_AVAILABLE:
        print("[worker][save] skipped: database unavailable")
        return

    print(
        f"[worker][save] start: history_rows={len(history)} transactions_rows={len(transactions)} "
        f"positions_rows={len(positions)} analyses={len(analyses)}"
    )

    conn = get_connection()
    c = conn.cursor()

    portfolio_value = float(top20_manager.portfolio_value())
    portfolio_time = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO portfolio (time, value) VALUES (%s, %s)",
        (portfolio_time, portfolio_value),
    )

    if not transactions.empty:
        for _, t in transactions.iterrows():
            c.execute(
                "INSERT INTO transactions (time, ticker, action, shares, price) VALUES (%s, %s, %s, %s, %s)",
                (t["time"], t["ticker"], t["action"], float(t["shares"]), float(t["price"])),
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
                    float(p["shares"]),
                    float(p["avg_cost"]),
                    float(p["current_price"]),
                    float(p["market_value"]),
                    float(p["allocation"]),
                    float(p["pnl"]),
                    float(p["pnl_pct"]),
                ),
            )

    if analyses:
        c.execute("DELETE FROM recommendation_signals")
        for a in analyses:
            c.execute(
                "INSERT INTO recommendation_signals (time, ticker, decision, score, price) VALUES (%s, %s, %s, %s, %s)",
                (
                    datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S"),
                    a["ticker"],
                    a["decision"],
                    float(a["score"]),
                    float(a["price"]),
                ),
            )

    conn.commit()
    c.close()
    conn.close()
    print("[worker][save] complete")


print("GitHub Actions worker starting one execution cycle...")


def build_run_key(now=None):
    """Create a stable run key to avoid duplicate trades from cron retries.

    We use 30-minute buckets because the workflow is typically scheduled every 30 minutes.
    """
    current = now or datetime.now(ZoneInfo("Europe/Paris"))
    bucket_minute = (current.minute // 30) * 30
    return f"{current.strftime('%Y-%m-%d')}-{current.hour:02d}-{bucket_minute:02d}"


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def restore_manager_state_from_db():
    if not DB_AVAILABLE:
        print("[worker][restore] skipped: database unavailable")
        return

    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT value FROM portfolio ORDER BY time DESC LIMIT 1")
    row = c.fetchone()
    latest_portfolio_value = float(row[0]) if row else None

    c.execute(
        """
        SELECT ticker, shares, avg_cost, current_price
        FROM position_snapshots
        WHERE time = (SELECT MAX(time) FROM position_snapshots)
        """
    )
    positions = c.fetchall() or []

    c.close()
    conn.close()

    if latest_portfolio_value is None or not positions:
        print("[worker][restore] no prior portfolio state found; using default manager state")
        return

    holdings = {}
    last_prices = {}
    market_value_total = 0.0
    for ticker, shares, avg_cost, current_price in positions:
        shares_f = float(shares)
        avg_cost_f = float(avg_cost)
        price_f = float(current_price)
        holdings[str(ticker)] = {
            "shares": shares_f,
            "avg_cost": avg_cost_f,
            "peak_price": max(price_f, avg_cost_f),
        }
        last_prices[str(ticker)] = price_f
        market_value_total += shares_f * price_f

    restored_cash = max(float(latest_portfolio_value) - market_value_total, 0.0)

    top20_manager.holdings = holdings
    top20_manager.last_price_by_ticker = last_prices
    top20_manager.cash = restored_cash
    top20_manager.transactions = []

    print(
        f"[worker][restore] restored holdings={len(holdings)} "
        f"cash={restored_cash:.2f} portfolio_value={latest_portfolio_value:.2f}"
    )


def main():
    event_name = os.getenv("GITHUB_EVENT_NAME", "").strip().lower() or "local"
    print(f"[worker][start] event={event_name} trade_mode={TRADE_MODE}")

    if DB_AVAILABLE:
        force_run = _truthy(os.getenv("FORCE_WORKER_RUN", "0"))
        skip_dedupe = event_name == "workflow_dispatch" or force_run

        if skip_dedupe:
            print(f"[worker][dedupe] bypassed: event={event_name} force_run={force_run}")
        else:
            run_key = build_run_key()
            print(f"[worker][dedupe] checking run_key={run_key}")
            if not claim_worker_run(run_key):
                print(f"[worker][dedupe] skip: run_key={run_key} already processed")
                return
            print(f"[worker][dedupe] acquired run_key={run_key}")
    else:
        print("[worker][dedupe] skipped: database unavailable")

    restore_manager_state_from_db()

    print("[worker][cycle] starting analysis cycle")
    history, transactions, positions, analyses = run_top20_cycle_with_signals()
    print(
        f"[worker][cycle] complete: history_rows={len(history)}, transactions_rows={len(transactions)}, "
        f"positions_rows={len(positions)}, analyses={len(analyses)}"
    )
    if analyses:
        sample = analyses[:5]
        print(f"[worker][cycle] sample analyses (up to 5): {sample}")
    else:
        print("[worker][cycle] no analyses generated")

    save(history, transactions, positions, analyses)
    print("[worker][done] cycle complete")


if __name__ == "__main__":
    main()
