from __future__ import annotations

import os
from datetime import datetime
from typing import Any
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.db import get_connection, init_db


app = FastAPI(title="kaibot-api", version="1.0.0")
logger = logging.getLogger("kaibot-api")

cors_origins = [x.strip() for x in os.getenv("API_CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ManualCheckRequest(BaseModel):
    ticker: str


def _fetch_rows(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


@app.on_event("startup")
def startup() -> None:
    # Do not fail process startup on DB cold-start/network hiccups.
    try:
        init_db()
    except Exception as exc:  # pragma: no cover
        logger.exception("startup init_db failed: %s", exc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/portfolio")
def get_portfolio(limit: int = 500) -> dict[str, Any]:
    rows = _fetch_rows("SELECT * FROM portfolio ORDER BY time DESC LIMIT %s", (max(min(limit, 5000), 1),))
    return {"rows": rows}


@app.get("/api/transactions")
def get_transactions(limit: int = 500) -> dict[str, Any]:
    rows = _fetch_rows("SELECT * FROM transactions ORDER BY time DESC LIMIT %s", (max(min(limit, 5000), 1),))
    return {"rows": rows}


@app.get("/api/positions")
def get_positions() -> dict[str, Any]:
    rows = _fetch_rows(
        """
        SELECT *
        FROM position_snapshots
        WHERE time = (SELECT MAX(time) FROM position_snapshots)
        ORDER BY market_value DESC
        """
    )
    return {"rows": rows}


@app.get("/api/signals")
def get_signals(limit: int = 50) -> dict[str, Any]:
    rows = _fetch_rows("SELECT * FROM recommendation_signals ORDER BY time DESC LIMIT %s", (max(min(limit, 500), 1),))
    return {"rows": rows}


@app.get("/api/goal/latest")
def get_goal_latest() -> dict[str, Any]:
    rows = _fetch_rows("SELECT * FROM agent_goal_snapshots ORDER BY ts DESC LIMIT 1")
    return {"row": rows[0] if rows else None}


@app.get("/api/manual-checks")
def get_manual_checks(limit: int = 20) -> dict[str, Any]:
    rows = _fetch_rows(
        "SELECT * FROM manual_ticker_checks ORDER BY added_at DESC NULLS LAST, id DESC LIMIT %s",
        (max(min(limit, 200), 1),),
    )
    return {"rows": rows}


@app.post("/api/manual-checks")
def add_manual_check(payload: ManualCheckRequest) -> dict[str, Any]:
    ticker = str(payload.ticker or "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    from src.advisor import run_manual_ticker_check

    rec = run_manual_ticker_check(ticker)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO manual_ticker_checks (ticker, decision, reason, score, price, signal_confidence, time, added_at, last_checked_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (ticker)
        DO UPDATE SET
            decision = EXCLUDED.decision,
            reason = EXCLUDED.reason,
            score = EXCLUDED.score,
            price = EXCLUDED.price,
            signal_confidence = EXCLUDED.signal_confidence,
            time = EXCLUDED.time,
            last_checked_at = NOW()
        """,
        (
            ticker,
            str(rec.get("decision", "HOLD")),
            ",".join(rec.get("decision_reasons", []) or [])[:1000],
            float(rec.get("score", 0.0)),
            float(rec.get("price", 0.0)),
            float(rec.get("signal_confidence", 0.0)),
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True, "recommendation": rec}


@app.delete("/api/manual-checks/{ticker}")
def delete_manual_check(ticker: str) -> dict[str, Any]:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="ticker is required")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM manual_ticker_checks WHERE ticker = %s", (symbol,))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}


@app.post("/api/cycle/run")
def run_cycle() -> dict[str, Any]:
    from src.advisor import run_top20_cycle_with_signals
    from worker.auto_worker import save as persist_cycle

    history, transactions, positions, analyses = run_top20_cycle_with_signals()
    persist_cycle(history, transactions, positions, analyses)
    return {
        "ok": True,
        "counts": {
            "history": int(len(history)),
            "transactions": int(len(transactions)),
            "positions": int(len(positions)),
            "analyses": int(len(analyses)),
            "buy": int(sum(1 for a in analyses if str(a.get("decision", "")).upper() == "BUY")),
            "sell": int(sum(1 for a in analyses if str(a.get("decision", "")).upper() == "SELL")),
            "hold": int(sum(1 for a in analyses if str(a.get("decision", "")).upper() == "HOLD")),
        },
    }
