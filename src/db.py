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
            price REAL
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
        CREATE TABLE IF NOT EXISTS backtest_sweep_results (
            run_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            rank INTEGER,
            buy_threshold REAL,
            sell_threshold REAL,
            min_buy_score REAL,
            slippage_bps REAL,
            fee_bps REAL,
            ending_value REAL,
            total_return_pct REAL,
            benchmark_return_pct REAL,
            excess_return_pct REAL,
            max_drawdown_pct REAL,
            volatility_pct REAL,
            sharpe_like REAL,
            turnover_ratio REAL,
            num_trades INTEGER
        );
    """
    )


    # Migration for fractional shares support.
    cur.execute("ALTER TABLE transactions ALTER COLUMN shares TYPE REAL USING shares::REAL;")
    cur.execute("ALTER TABLE position_snapshots ALTER COLUMN shares TYPE REAL USING shares::REAL;")

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


def save_backtest_sweep_results(run_id, rows):
    if not rows:
        return

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM backtest_sweep_results WHERE run_id = %s", (run_id,))

    for row in rows:
        cur.execute(
            """
            INSERT INTO backtest_sweep_results (
                run_id, rank, buy_threshold, sell_threshold, min_buy_score,
                slippage_bps, fee_bps, ending_value, total_return_pct,
                benchmark_return_pct, excess_return_pct, max_drawdown_pct,
                volatility_pct, sharpe_like, turnover_ratio, num_trades
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(run_id),
                int(row.get("rank", 0)),
                float(row.get("buy_threshold", 0.0)),
                float(row.get("sell_threshold", 0.0)),
                float(row.get("min_buy_score", 0.0)),
                float(row.get("slippage_bps", 0.0)),
                float(row.get("fee_bps", 0.0)),
                float(row.get("ending_value", 0.0)),
                float(row.get("total_return_pct", 0.0)),
                float(row.get("benchmark_return_pct", 0.0)),
                float(row.get("excess_return_pct", 0.0)),
                float(row.get("max_drawdown_pct", 0.0)),
                float(row.get("volatility_pct", 0.0)),
                float(row.get("sharpe_like", 0.0)),
                float(row.get("turnover_ratio", 0.0)),
                int(row.get("num_trades", 0)),
            ),
        )

    conn.commit()
    cur.close()
    conn.close()
