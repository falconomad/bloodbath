import os

# API Keys
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not ALPACA_API_KEY or not ALPACA_API_SECRET or not GEMINI_API_KEY:
    print("Missing API keys. Please set ALPACA_API_KEY, ALPACA_API_SECRET, and GEMINI_API_KEY.")
    exit(1)

# Global Settings
PAPER_TRADING = True
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

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

# Free Tier Rate Limits (Gemini Flash allows 15 Requests Per Minute)
# We make 3 calls per ticker, so we need to spread them out.
API_SLEEP_SECONDS = int(os.environ.get("API_SLEEP_SECONDS", 15)) # spacing between Gemini calls
GEMINI_MIN_SECONDS_BETWEEN_CALLS = int(os.environ.get("GEMINI_MIN_SECONDS_BETWEEN_CALLS", 15))
GEMINI_MAX_RETRIES_ON_429 = int(os.environ.get("GEMINI_MAX_RETRIES_ON_429", 1))
GEMINI_QUOTA_STATE_PATH = os.environ.get("GEMINI_QUOTA_STATE_PATH", "logs/gemini_quota_state.json")
MACRO_DOWNTURN_LIMIT = -2.0 # Halt buying if SPY is down > 2%
