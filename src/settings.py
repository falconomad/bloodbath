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


def _trade_mode_from_env():
    mode = _str_from_env("TRADE_MODE", "NORMAL").strip().upper()
    if mode not in {"NORMAL", "AGGRESSIVE"}:
        return "NORMAL"
    return mode


TRADE_MODE = _trade_mode_from_env()
TOP20_STARTING_CAPITAL = _float_from_env("TOP20_STARTING_CAPITAL", 2000)

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

# Fetch pacing for free-tier data providers.
FETCH_BATCH_SIZE = _int_from_env("FETCH_BATCH_SIZE", 20)
