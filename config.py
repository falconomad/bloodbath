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
MAX_ALLOCATION_PCT = 20  # Never allocate more than 20% of buying power to a single ticker
PROFIT_TAKE_PCT = 1.5    # Strict exit at 1.5% profit
STOP_LOSS_PCT = -1.0     # Strict exit at -1.0% loss
MACRO_DOWNTURN_LIMIT = -2.0 # Halt buying if SPY is down > 2%
