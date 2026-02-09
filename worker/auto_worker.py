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
from src.advisor import run_top20_cycle_with_signals
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
