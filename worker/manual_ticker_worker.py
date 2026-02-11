from pathlib import Path
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db import get_connection, init_db
from src.advisor import run_manual_ticker_check
from src.settings import MANUAL_CHECK_TICKER

DB_AVAILABLE = True
MAX_TRACKED_TICKERS = 5

try:
    init_db()
except Exception as exc:
    DB_AVAILABLE = False
    print(f"[manual-worker] database initialization failed: {exc}")


def _resolve_requested_ticker() -> str:
    cli = os.getenv("MANUAL_CHECK_TICKER_INPUT", "").strip().upper()
    if cli:
        return cli.replace(".", "-")
    return str(MANUAL_CHECK_TICKER or "").strip().upper().replace(".", "-")


def _short_reason(reasons):
    if not reasons:
        return "-"
    first = str((reasons or ["-"])[0])
    first = first.replace("guardrail:", "").replace("veto:", "").replace("stability:", "")
    first = first.replace("confidence:", "").replace("volatility:", "").replace("portfolio_risk:", "")
    return first[:120] if first else "-"


def _fetch_tracked_rows(conn):
    c = conn.cursor()
    c.execute(
        """
        SELECT id, ticker, added_at
        FROM manual_ticker_checks
        ORDER BY added_at ASC NULLS LAST, id ASC
        """
    )
    rows = c.fetchall() or []
    c.close()
    return rows


def _evaluate_ticker(ticker: str):
    result = run_manual_ticker_check(ticker)
    if result.get("error"):
        return {
            "ticker": ticker,
            "decision": "ERROR",
            "reason": str(result.get("error")),
            "score": 0.0,
            "price": 0.0,
            "signal_confidence": 0.0,
        }
    return {
        "ticker": str(result.get("ticker", ticker)).upper(),
        "decision": str(result.get("decision", "HOLD")),
        "reason": _short_reason(result.get("decision_reasons", [])),
        "score": float(result.get("score", 0.0)),
        "price": float(result.get("price", 0.0)),
        "signal_confidence": float(result.get("signal_confidence", 0.0)),
    }


def _upsert_result(conn, payload: dict, now_ts_text: str):
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO manual_ticker_checks (
            time, ticker, decision, reason, score, price, signal_confidence, added_at, last_checked_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (ticker)
        DO UPDATE SET
            time = EXCLUDED.time,
            decision = EXCLUDED.decision,
            reason = EXCLUDED.reason,
            score = EXCLUDED.score,
            price = EXCLUDED.price,
            signal_confidence = EXCLUDED.signal_confidence,
            last_checked_at = NOW()
        """,
        (
            now_ts_text,
            str(payload.get("ticker", "")).upper(),
            str(payload.get("decision", "HOLD")),
            str(payload.get("reason", "-")),
            float(payload.get("score", 0.0)),
            float(payload.get("price", 0.0)),
            float(payload.get("signal_confidence", 0.0)),
        ),
    )
    c.close()


def _delete_ticker_row(conn, ticker: str):
    c = conn.cursor()
    c.execute("DELETE FROM manual_ticker_checks WHERE ticker = %s", (str(ticker).upper(),))
    c.close()


def _replace_oldest_ticker(conn, old_ticker: str, new_ticker: str, payload: dict, now_ts_text: str):
    c = conn.cursor()
    c.execute(
        """
        UPDATE manual_ticker_checks
        SET
            ticker = %s,
            time = %s,
            decision = %s,
            reason = %s,
            score = %s,
            price = %s,
            signal_confidence = %s,
            added_at = NOW(),
            last_checked_at = NOW()
        WHERE ticker = %s
        """,
        (
            new_ticker,
            now_ts_text,
            str(payload.get("decision", "HOLD")),
            str(payload.get("reason", "-")),
            float(payload.get("score", 0.0)),
            float(payload.get("price", 0.0)),
            float(payload.get("signal_confidence", 0.0)),
            old_ticker,
        ),
    )
    c.close()


def main():
    if not DB_AVAILABLE:
        print("[manual-worker] database unavailable")
        return

    requested_ticker = _resolve_requested_ticker()
    conn = get_connection()

    try:
        tracked_rows = _fetch_tracked_rows(conn)
        tracked_tickers = [str(r[1]).upper() for r in tracked_rows if str(r[1]).strip()]

        if requested_ticker:
            if requested_ticker not in tracked_tickers:
                if len(tracked_tickers) >= MAX_TRACKED_TICKERS:
                    oldest_ticker = tracked_tickers[0]
                    print(
                        f"[manual-worker] watchlist full ({MAX_TRACKED_TICKERS}); "
                        f"replacing oldest {oldest_ticker} -> {requested_ticker}"
                    )
                    now_ts_text = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S")
                    new_payload = _evaluate_ticker(requested_ticker)
                    _replace_oldest_ticker(conn, oldest_ticker, requested_ticker, new_payload, now_ts_text)
                    tracked_tickers = tracked_tickers[1:] + [requested_ticker]
                else:
                    tracked_tickers.append(requested_ticker)
            else:
                print(f"[manual-worker] requested ticker already tracked: {requested_ticker}")

        if not tracked_tickers:
            print("[manual-worker] no tracked tickers. Set MANUAL_CHECK_TICKER or pass workflow input ticker")
            conn.commit()
            return

        print(f"[manual-worker] refreshing tracked tickers: {tracked_tickers}")
        now_ts_text = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S")
        for ticker in tracked_tickers:
            payload = _evaluate_ticker(ticker)
            if str(payload.get("decision", "")).upper() == "ERROR":
                _delete_ticker_row(conn, ticker)
                print(f"[manual-worker] {ticker} removed from watchlist (decision=ERROR reason={payload['reason']})")
                continue
            _upsert_result(conn, payload, now_ts_text)
            print(
                "[manual-worker] "
                f"{ticker} decision={payload['decision']} score={payload['score']:.4f} "
                f"conf={payload['signal_confidence']:.4f} reason={payload['reason']}"
            )

        conn.commit()
        print("[manual-worker] save complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
