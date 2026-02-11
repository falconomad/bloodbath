import os


def _float_from_env(name, default):
    raw = os.getenv(name)
    if raw is None:
        return float(default)

    try:
        return float(raw)
    except ValueError:
        return float(default)


def _int_from_env(name, default):
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _str_from_env(name, default):
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    return str(raw)


def _bool_from_env(name, default):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _trade_mode_from_env():
    mode = _str_from_env("TRADE_MODE", "NORMAL").strip().upper()
    if mode not in {"NORMAL", "AGGRESSIVE"}:
        return "NORMAL"
    return mode


TRADE_MODE = _trade_mode_from_env()
STARTING_CAPITAL = _float_from_env("STARTING_CAPITAL", _float_from_env("TOP20_STARTING_CAPITAL", 2000))

# Signal thresholds for top20 cycle decisions.
SIGNAL_BUY_THRESHOLD = _float_from_env("SIGNAL_BUY_THRESHOLD", 1.0 if TRADE_MODE == "NORMAL" else 0.55)
SIGNAL_SELL_THRESHOLD = _float_from_env("SIGNAL_SELL_THRESHOLD", -1.0 if TRADE_MODE == "NORMAL" else -0.7)

# Recommendation threshold used in generate_recommendation metadata.
RECOMMENDATION_DECISION_THRESHOLD = _float_from_env(
    "RECOMMENDATION_DECISION_THRESHOLD", 0.7 if TRADE_MODE == "NORMAL" else 0.45
)

# Trading engine knobs.
TOP20_MIN_BUY_SCORE = _float_from_env("TOP20_MIN_BUY_SCORE", 0.75 if TRADE_MODE == "NORMAL" else 0.45)
TOP20_SLIPPAGE_BPS = _float_from_env("TOP20_SLIPPAGE_BPS", 5.0)
TOP20_FEE_BPS = _float_from_env("TOP20_FEE_BPS", 1.0)
TOP20_TAKE_PROFIT_PCT = _float_from_env("TOP20_TAKE_PROFIT_PCT", 0.30 if TRADE_MODE == "NORMAL" else 0.08)

# Fetch pacing for free-tier data providers.
FETCH_BATCH_SIZE = _int_from_env("FETCH_BATCH_SIZE", 20)

# Optional execution aggressiveness control (default off for safety).
ENABLE_POSITION_ROTATION = _bool_from_env("ENABLE_POSITION_ROTATION", False)

# Optional manual ticker probe shown as a separate dashboard section.
MANUAL_CHECK_TICKER = _str_from_env("MANUAL_CHECK_TICKER", "").strip().upper()

# Execution safeguards.
EXECUTION_MAX_CONSECUTIVE_FAILURES = _int_from_env("EXECUTION_MAX_CONSECUTIVE_FAILURES", 3)
EXECUTION_STALE_DATA_MAX_AGE_HOURS = _float_from_env("EXECUTION_STALE_DATA_MAX_AGE_HOURS", 72.0)
EXECUTION_ANOMALY_ZSCORE_THRESHOLD = _float_from_env("EXECUTION_ANOMALY_ZSCORE_THRESHOLD", 4.0)
EXECUTION_MAX_CYCLE_NOTIONAL_TURNOVER = _float_from_env("EXECUTION_MAX_CYCLE_NOTIONAL_TURNOVER", 1.25)
EXECUTION_STEP_MAX_RETRIES = _int_from_env("EXECUTION_STEP_MAX_RETRIES", 2)

# Optional Gemini pre-trade guard (budgeted).
GEMINI_API_KEY = _str_from_env("GEMINI_API_KEY", "").strip()
ENABLE_GEMINI_PRETRADE_CHECK = _bool_from_env("ENABLE_GEMINI_PRETRADE_CHECK", True)
GEMINI_MODEL = _str_from_env("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
GEMINI_MAX_CALLS_PER_CYCLE = _int_from_env("GEMINI_MAX_CALLS_PER_CYCLE", 1)
GEMINI_MAX_CALLS_PER_DAY = _int_from_env("GEMINI_MAX_CALLS_PER_DAY", 80)
GEMINI_TIMEOUT_SECONDS = _float_from_env("GEMINI_TIMEOUT_SECONDS", 8.0)

# Loser-hunter aggressive candidate mode.
LOSER_HUNTER_ENABLED = _bool_from_env("LOSER_HUNTER_ENABLED", True)
LOSER_UNIVERSE_SIZE = _int_from_env("LOSER_UNIVERSE_SIZE", 500)
LOSER_TOP_K = _int_from_env("LOSER_TOP_K", 20)
LOSER_MIN_DROP_PCT = _float_from_env("LOSER_MIN_DROP_PCT", 0.06)
LOSER_REQUIRE_STABILIZATION = _bool_from_env("LOSER_REQUIRE_STABILIZATION", True)
LOSER_MIN_DIP_SCORE = _float_from_env("LOSER_MIN_DIP_SCORE", 0.05)
LOSER_INCLUDE_GAINERS = _bool_from_env("LOSER_INCLUDE_GAINERS", False)
LOSER_PROFIT_ALERT_PCT = _float_from_env("LOSER_PROFIT_ALERT_PCT", 0.05)
