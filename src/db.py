import os

import psycopg2


def get_connection():
    """
    Creates and returns a PostgreSQL connection using DATABASE_URL
    environment variable.
    """
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise Exception("DATABASE_URL environment variable is not set")

    return psycopg2.connect(db_url)


def init_db():
    """
    Initializes required tables if they do not already exist.
    Safe to call multiple times.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            time TEXT,
            value REAL
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            time TEXT,
            ticker TEXT,
            action TEXT,
            shares INTEGER,
            price REAL
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS position_snapshots (
            time TEXT,
            ticker TEXT,
            shares INTEGER,
            avg_cost REAL,
            current_price REAL,
            market_value REAL,
            allocation REAL,
            pnl REAL,
            pnl_pct REAL
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS worker_runs (
            run_key TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """
    )

    conn.commit()
    cur.close()
    conn.close()


def claim_worker_run(run_key):
    """Return True if this run_key was newly claimed, False if already processed."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO worker_runs (run_key)
        VALUES (%s)
        ON CONFLICT (run_key) DO NOTHING
        RETURNING run_key;
        """,
        (run_key,),
    )

    claimed = cur.fetchone() is not None
    conn.commit()
    cur.close()
    conn.close()
    return claimed
