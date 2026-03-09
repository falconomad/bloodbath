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

# Hardcoded Trading Rules & Risk Management limits
MAX_ALLOCATION_PCT = 20 # Never put more than 20% of buying power into a single trade
PROFIT_TAKE_PCT = 5.0 # Pre-emptive sell if unrealized P/L is over +5%
STOP_LOSS_PCT = -3.0 # Pre-emptive exit if unrealized P/L drops below -3%

# Free Tier Rate Limits (Gemini Flash allows 15 Requests Per Minute)
# We make 3 calls per ticker, so we need to spread them out.
API_SLEEP_SECONDS = 15 # 15s per loop iteration = 4 requests per minute = safess
MACRO_DOWNTURN_LIMIT = -2.0 # Halt buying if SPY is down > 2%
