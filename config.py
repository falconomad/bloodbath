import os


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


# API Keys
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not ALPACA_API_KEY or not ALPACA_API_SECRET or not GEMINI_API_KEY:
    print("Missing API keys. Please set ALPACA_API_KEY, ALPACA_API_SECRET, and GEMINI_API_KEY.")
    exit(1)

# Global Settings
PAPER_TRADING = True

# Hardcoded Trading Rules & Risk Management limits
MAX_ALLOCATION_PCT = 20 # Never put more than 20% of buying power into a single trade
PROFIT_TAKE_PCT = 5.0 # Pre-emptive sell if unrealized P/L is over +5%
STOP_LOSS_PCT = -3.0 # Pre-emptive exit if unrealized P/L drops below -3%

# Market Filter Constraints (Pre-AI)
MIN_PRICE = 5.0 # Ignore penny stocks
MIN_DOLLAR_VOLUME = 10_000_000 # Ignore illiquid stocks
MAX_CANDIDATES = 5 # Send max 5 stocks strictly to the AI council to preserve quota
MAX_GEMINI_CALLS_PER_RUN = int(os.environ.get("MAX_GEMINI_CALLS_PER_RUN", 3))
PRE_FILTER_MIN_TECH_SCORE = int(os.environ.get("PRE_FILTER_MIN_TECH_SCORE", 55))
PRE_FILTER_MIN_SENTIMENT_SCORE = int(os.environ.get("PRE_FILTER_MIN_SENTIMENT_SCORE", 45))
PRE_FILTER_MIN_LOCAL_SCORE = int(os.environ.get("PRE_FILTER_MIN_LOCAL_SCORE", 60))
PRE_FILTER_MIN_CONFIDENCE = float(os.environ.get("PRE_FILTER_MIN_CONFIDENCE", 0.55))
RELAXED_SENTIMENT_DELTA = int(os.environ.get("RELAXED_SENTIMENT_DELTA", 10))
RELAXED_CONFIDENCE_DELTA = float(os.environ.get("RELAXED_CONFIDENCE_DELTA", 0.08))
MAX_RELAXED_CANDIDATES = int(os.environ.get("MAX_RELAXED_CANDIDATES", 1))
GEMINI_MINUTE_STEP = int(os.environ.get("GEMINI_MINUTE_STEP", 30))
ONLY_WHEN_MARKET_OPEN = _bool_env("ONLY_WHEN_MARKET_OPEN", True)

# Gemini Paid Tier Toggle
GEMINI_PAID_TIER = _bool_env("GEMINI_PAID_TIER", False)

# Gemini Rate Limit Guards
if GEMINI_PAID_TIER:
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    API_SLEEP_SECONDS = int(os.environ.get("API_SLEEP_SECONDS", 1))
    GEMINI_MIN_SECONDS_BETWEEN_CALLS = int(os.environ.get("GEMINI_MIN_SECONDS_BETWEEN_CALLS", 0))
    GEMINI_DAILY_LIMIT = int(os.environ.get("GEMINI_DAILY_LIMIT", 5000))
else:
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    API_SLEEP_SECONDS = int(os.environ.get("API_SLEEP_SECONDS", 15))
    GEMINI_MIN_SECONDS_BETWEEN_CALLS = int(os.environ.get("GEMINI_MIN_SECONDS_BETWEEN_CALLS", 15))
    GEMINI_DAILY_LIMIT = int(os.environ.get("GEMINI_DAILY_LIMIT", 18))

GEMINI_MAX_RETRIES_ON_429 = int(os.environ.get("GEMINI_MAX_RETRIES_ON_429", 1))
GEMINI_QUOTA_STATE_PATH = os.environ.get("GEMINI_QUOTA_STATE_PATH", "logs/gemini_quota_state.json")
ENGINE_EVENTS_PATH = os.environ.get("ENGINE_EVENTS_PATH", "logs/engine_events.jsonl")
PROFIT_SUMMARY_PATH = os.environ.get("PROFIT_SUMMARY_PATH", "logs/profit_summary.json")
BASELINE_EQUITY = _float_env("BASELINE_EQUITY", 0.0)
MACRO_DOWNTURN_LIMIT = -2.0 # Halt buying if SPY is down > 2%

# Day Trading & Safety Controls
TRAILING_STOP_PCT = _float_env("TRAILING_STOP_PCT", 2.5) # Sell if position drops 2.5% from its peak price
EOD_LIQUIDATION_MINUTES = int(os.environ.get("EOD_LIQUIDATION_MINUTES", 15)) # Close all positions 15 min before market close
POSITION_PEAKS_PATH = os.environ.get("POSITION_PEAKS_PATH", "logs/position_peaks.json")
