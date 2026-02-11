#!/usr/bin/env python3
"""Reset persisted trading state and set a new starting capital."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import psycopg2
from zoneinfo import ZoneInfo

RESET_TABLES = [
    "portfolio",
    "transactions",
    "position_snapshots",
    "recommendation_signals",
    "recommendation_trace",
    "worker_runs",
    "decision_memory",
    "decision_engine_meta",
    "manual_ticker_checks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh-start the engine with a new starting capital amount.",
    )
    parser.add_argument(
        "capital",
        type=float,
        help="Starting capital in USD (for example: 5000)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file to update STARTING_CAPITAL (default: .env)",
    )
    parser.add_argument(
        "--skip-env-update",
        action="store_true",
        help="Do not modify env file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without applying changes.",
    )
    return parser.parse_args()


def validate_capital(capital: float) -> float:
    if capital <= 0:
        raise ValueError("capital must be greater than 0")
    return round(float(capital), 2)


def update_env_file(env_path: Path, capital: float) -> None:
    line = f"STARTING_CAPITAL={capital:.2f}"
    if not env_path.exists():
        env_path.write_text(line + "\n", encoding="utf-8")
        return

    raw = env_path.read_text(encoding="utf-8")
    primary_pattern = re.compile(r"^\s*STARTING_CAPITAL\s*=.*$", re.MULTILINE)
    legacy_pattern = re.compile(r"^\s*TOP20_STARTING_CAPITAL\s*=.*$", re.MULTILINE)
    if primary_pattern.search(raw):
        updated = primary_pattern.sub(line, raw)
    elif legacy_pattern.search(raw):
        updated = legacy_pattern.sub(line, raw)
    else:
        suffix = "" if raw.endswith("\n") or not raw else "\n"
        updated = f"{raw}{suffix}{line}\n"
    env_path.write_text(updated, encoding="utf-8")


def reset_database(capital: float, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[dry-run] would truncate tables: {', '.join(RESET_TABLES)}")
        print(f"[dry-run] would seed portfolio with value={capital:.2f}")
        return

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    for table in RESET_TABLES:
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")

    ts = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO portfolio (time, value) VALUES (%s, %s)", (ts, capital))

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    args = parse_args()
    capital = validate_capital(args.capital)

    if args.dry_run:
        print(f"[dry-run] fresh start capital={capital:.2f}")
    else:
        print(f"[fresh-start] resetting state with capital={capital:.2f}")

    reset_database(capital=capital, dry_run=args.dry_run)

    if args.skip_env_update:
        msg = "[fresh-start] skipped env update"
        print(msg if not args.dry_run else msg.replace("[fresh-start]", "[dry-run]"))
    else:
        env_file = Path(args.env_file)
        if args.dry_run:
            print(f"[dry-run] would update {env_file} with STARTING_CAPITAL={capital:.2f}")
        else:
            update_env_file(env_file, capital)
            print(f"[fresh-start] updated {env_file} -> STARTING_CAPITAL={capital:.2f}")

    if not args.dry_run:
        print("[fresh-start] done")
