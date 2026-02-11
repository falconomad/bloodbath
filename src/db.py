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
            shares REAL,
            price REAL
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS position_snapshots (
            time TEXT,
            ticker TEXT,
            shares REAL,
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
        CREATE TABLE IF NOT EXISTS recommendation_signals (
            time TEXT,
            ticker TEXT,
            decision TEXT,
            score REAL,
            price REAL,
            mover_bucket TEXT,
            daily_return REAL
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_trace (
            id BIGSERIAL PRIMARY KEY,
            ts TEXT,
            ticker TEXT,
            payload_json TEXT
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

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_memory (
            ticker TEXT PRIMARY KEY,
            last_decision TEXT,
            last_cycle INTEGER,
            last_flip_cycle INTEGER,
            last_non_hold_cycle INTEGER,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_engine_meta (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_ticker_checks (
            id BIGSERIAL PRIMARY KEY,
            ticker TEXT,
            decision TEXT,
            reason TEXT,
            score REAL,
            price REAL,
            signal_confidence REAL,
            added_at TIMESTAMPTZ DEFAULT NOW(),
            last_checked_at TIMESTAMPTZ DEFAULT NOW()
        );
    """
    )


    # Migration for fractional shares support.
    cur.execute("ALTER TABLE transactions ALTER COLUMN shares TYPE REAL USING shares::REAL;")
    cur.execute("ALTER TABLE position_snapshots ALTER COLUMN shares TYPE REAL USING shares::REAL;")
    cur.execute("ALTER TABLE recommendation_signals ADD COLUMN IF NOT EXISTS mover_bucket TEXT;")
    cur.execute("ALTER TABLE recommendation_signals ADD COLUMN IF NOT EXISTS daily_return REAL;")
    cur.execute("ALTER TABLE manual_ticker_checks ADD COLUMN IF NOT EXISTS added_at TIMESTAMPTZ DEFAULT NOW();")
    cur.execute("ALTER TABLE manual_ticker_checks ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMPTZ DEFAULT NOW();")
    cur.execute(
        """
        ALTER TABLE manual_ticker_checks
        ADD COLUMN IF NOT EXISTS time TEXT;
        """
    )
    cur.execute("UPDATE manual_ticker_checks SET ticker = UPPER(COALESCE(ticker, ''));")
    cur.execute(
        """
        DELETE FROM manual_ticker_checks a
        USING manual_ticker_checks b
        WHERE a.id < b.id
          AND UPPER(COALESCE(a.ticker, '')) = UPPER(COALESCE(b.ticker, ''));
        """
    )
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS manual_ticker_checks_ticker_uq ON manual_ticker_checks (ticker);")

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
