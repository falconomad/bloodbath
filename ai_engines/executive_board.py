import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

_quota_state = {
    "date": "",
    "daily_calls": 0,
    "last_call_ts": 0.0,
}


def _load_quota_state():
    global _quota_state
    path = config.GEMINI_QUOTA_STATE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _quota_state.update(data)
    except Exception:
        pass


def _save_quota_state():
    path = config.GEMINI_QUOTA_STATE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_quota_state, f)


def _today_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _rate_limited_generate(prompt: str):
    _load_quota_state()
    today = _today_utc()
    if _quota_state.get("date") != today:
        _quota_state["date"] = today
        _quota_state["daily_calls"] = 0

    # Respect free-tier day cap guard before calling provider.
    if int(_quota_state.get("daily_calls", 0)) >= 18:
        raise RuntimeError("Gemini daily safety budget reached.")

    # Ensure spacing between calls.
    now = time.time()
    elapsed = now - float(_quota_state.get("last_call_ts", 0.0))
    min_gap = max(int(config.GEMINI_MIN_SECONDS_BETWEEN_CALLS), 0)
    if elapsed < min_gap:
        time.sleep(min_gap - elapsed)

    retries = max(int(config.GEMINI_MAX_RETRIES_ON_429), 0)
    attempt = 0
    while True:
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
            )
            _quota_state["daily_calls"] = int(_quota_state.get("daily_calls", 0)) + 1
            _quota_state["last_call_ts"] = time.time()
            _save_quota_state()
            return response
        except Exception as exc:
            msg = str(exc)
            is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if not is_429 or attempt >= retries:
                raise
            # Parse "retry in Xs" hints when present.
            match = re.search(r"retry in ([0-9]+(?:\\.[0-9]+)?)s", msg.lower())
            wait_s = float(match.group(1)) if match else 60.0
            wait_s = max(wait_s, float(config.GEMINI_MIN_SECONDS_BETWEEN_CALLS))
            logging.warning("[ExecutiveBoard] Gemini 429; sleeping %.2fs before retry", wait_s)
            time.sleep(wait_s)
            attempt += 1


def _deterministic_fallback(symbol, current_positions, technical_report, sentiment_report, reason):
    held = next((p for p in current_positions if p.get("symbol") == symbol), None)
    tech = int(technical_report.get("score", 0) or 0)
    sent = int(sentiment_report.get("score", 50) or 50)
    if held:
        pnl = float(held.get("unrealized_pl_pct", 0.0))
        if pnl > config.PROFIT_TAKE_PCT or pnl < config.STOP_LOSS_PCT:
            return {
                "symbol": symbol,
                "action": "sell",
                "allocation_pct": 100,
                "chain_of_thought": f"Fallback deterministic exit trigger. {reason}",
            }
    if tech > 70 and sent > 70:
        alloc = 60 if (tech > 85 and sent > 85) else 40
        return {
            "symbol": symbol,
            "action": "buy",
            "allocation_pct": alloc,
            "chain_of_thought": f"Fallback deterministic buy due to strong dual conviction. {reason}",
        }
    return {
        "symbol": symbol,
        "action": "hold",
        "allocation_pct": 0,
        "chain_of_thought": f"Fallback deterministic hold. {reason}",
    }

def make_final_decision(symbol, current_positions, buying_power, technical_report, sentiment_report):
    """
    Synthesizes reports from the Technical and Sentiment Analysts to output a final trade recommendation.
    """
    prompt = f"""
    You are the Chief Investment Officer (CIO) of an aggressive algorithmic day-trading fund.
    Your goal is to increase portfolio value today, never holding overnight.

    Asset: {symbol}
    Current Buying Power: ${buying_power}
    Open Positions: {json.dumps(current_positions)}

    --- ANALYST REPORTS ---
    Technical Analyst: {json.dumps(technical_report)}
    Sentiment Analyst: {json.dumps(sentiment_report)}
    -----------------------

    Rules for ABSOLUTE COMPLIANCE:
    1. STRICT EXITS (Profits): If {symbol} is currently held and 'unrealized_pl_pct' > {config.PROFIT_TAKE_PCT}%, you MUST recommend a "sell" action (allocation_pct 100) to lock in the increment.
    2. STRICT EXITS (Losses): If {symbol} is currently held and 'unrealized_pl_pct' drops below {config.STOP_LOSS_PCT}%, you MUST recommend a "sell" action to cut the loss.
    3. NEW ENTRIES: If we do not hold this stock and you have buying power, evaluate the Analyst Reports. You only want to buy stocks that have BOTH strong upward technical momentum (> 70) AND a bullish fundamental catalyst (> 70).
    4. POSITION SIZING: If buying, output the percentage of your 'Current Buying Power' to allocate to {symbol} (e.g. 50 meaning 50%). Note that the Risk Manager will cap you at {config.MAX_ALLOCATION_PCT}%, so output the raw conviction allocation you desire.

    Output JSON ONLY:
    {{
        "chain_of_thought": "Briefly synthesize the two analyst reports and justify the final execution decision based on the firm's strict rules.",
        "action": "buy" | "sell" | "hold",
        "allocation_pct": <integer 0-100>
    }}
    """
    
    try:
        response = _rate_limited_generate(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text)
        data['symbol'] = symbol
        return data
    except Exception as e:
        logging.error(f"[ExecutiveBoard] Error synthesizing decision for {symbol}: {e}", exc_info=True)
        return _deterministic_fallback(
            symbol=symbol,
            current_positions=current_positions,
            technical_report=technical_report,
            sentiment_report=sentiment_report,
            reason=f"Executive engine failed: {e}",
        )


def make_batch_decisions(candidates, current_positions, buying_power):
    """
    candidates: list[{
      "symbol": str,
      "technical_report": {...},
      "sentiment_report": {...}
    }]
    Returns dict[symbol] -> decision payload.
    """
    if not candidates:
        return {}

    slim_candidates = []
    for c in candidates:
        symbol = str(c.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        tech = c.get("technical_report", {}) or {}
        sent = c.get("sentiment_report", {}) or {}
        # Compact payload to reduce token usage.
        slim_candidates.append(
            {
                "symbol": symbol,
                "tech_score": int(tech.get("score", 0) or 0),
                "tech_rationale": str(tech.get("rationale", ""))[:180],
                "sent_score": int(sent.get("score", 50) or 50),
                "sent_rationale": str(sent.get("rationale", ""))[:180],
            }
        )

    prompt = f"""
    You are the Chief Investment Officer (CIO) of an aggressive algorithmic day-trading fund.
    Goal: increase portfolio value today, never hold overnight.

    Current Buying Power: ${buying_power}
    Open Positions: {json.dumps(current_positions)}
    Candidate Reports: {json.dumps(slim_candidates)}

    Rules for ABSOLUTE COMPLIANCE:
    1. STRICT EXITS (Profits): If a candidate is currently held and unrealized_pl_pct > {config.PROFIT_TAKE_PCT}%, action MUST be "sell" with allocation_pct 100.
    2. STRICT EXITS (Losses): If a candidate is currently held and unrealized_pl_pct < {config.STOP_LOSS_PCT}%, action MUST be "sell" with allocation_pct 100.
    3. NEW ENTRIES: Only "buy" when BOTH tech_score > 70 and sent_score > 70.
    4. POSITION SIZING: allocation_pct is raw conviction 0-100; risk manager will cap at {config.MAX_ALLOCATION_PCT}%.

    Output JSON ONLY with this exact schema:
    {{
      "decisions": [
        {{
          "symbol": "TICKER",
          "action": "buy" | "sell" | "hold",
          "allocation_pct": <integer 0-100>,
          "chain_of_thought": "single brief sentence"
        }}
      ]
    }}
    """

    try:
        response = _rate_limited_generate(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        payload = json.loads(text)
        decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
        out = {}
        for row in decisions:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).upper().strip()
            if not symbol:
                continue
            out[symbol] = {
                "symbol": symbol,
                "action": str(row.get("action", "hold")).lower(),
                "allocation_pct": int(max(0, min(100, int(row.get("allocation_pct", 0) or 0)))),
                "chain_of_thought": str(row.get("chain_of_thought", ""))[:600],
            }
        return out
    except Exception as e:
        logging.error("[ExecutiveBoard] Batch decision failed: %s", e, exc_info=True)
        out = {}
        for c in candidates:
            symbol = str(c.get("symbol", "")).upper().strip()
            if not symbol:
                continue
            out[symbol] = _deterministic_fallback(
                symbol=symbol,
                current_positions=current_positions,
                technical_report=c.get("technical_report", {}) or {},
                sentiment_report=c.get("sentiment_report", {}) or {},
                reason=f"Batch executive failed: {e}",
            )
        return out
