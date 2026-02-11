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

try:
    init_db()
except Exception as exc:
    DB_AVAILABLE = False
    print(f"[manual-worker] database initialization failed: {exc}")


def _resolve_ticker() -> str:
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


def save_manual_result(result: dict):
    if not DB_AVAILABLE:
        print("[manual-worker] save skipped: database unavailable")
        return

    conn = get_connection()
    c = conn.cursor()

    now_ts = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        """
        INSERT INTO manual_ticker_checks (
            time, ticker, decision, reason, score, price, signal_confidence
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            now_ts,
            str(result.get("ticker", "")),
            str(result.get("decision", "HOLD")),
            str(result.get("reason", "-")),
            float(result.get("score", 0.0)),
            float(result.get("price", 0.0)),
            float(result.get("signal_confidence", 0.0)),
        ),
    )

    conn.commit()
    c.close()
    conn.close()
    print(f"[manual-worker] saved check for {result.get('ticker', '')}")


def main():
    ticker = _resolve_ticker()
    if not ticker:
        print("[manual-worker] no ticker configured; set MANUAL_CHECK_TICKER or MANUAL_CHECK_TICKER_INPUT")
        return

    print(f"[manual-worker] running manual check for ticker={ticker}")
    result = run_manual_ticker_check(ticker)
    if result.get("error"):
        print(f"[manual-worker] check failed: {result['error']}")
        return

    payload = {
        "ticker": str(result.get("ticker", ticker)),
        "decision": str(result.get("decision", "HOLD")),
        "reason": _short_reason(result.get("decision_reasons", [])),
        "score": float(result.get("score", 0.0)),
        "price": float(result.get("price", 0.0)),
        "signal_confidence": float(result.get("signal_confidence", 0.0)),
    }
    print(
        "[manual-worker] decision="
        f"{payload['decision']} score={payload['score']:.4f} conf={payload['signal_confidence']:.4f} "
        f"reason={payload['reason']}"
    )
    save_manual_result(payload)


if __name__ == "__main__":
    main()
