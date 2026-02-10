from __future__ import annotations

from functools import lru_cache


TICKER_DOMAIN = {
    "AAPL": "apple.com",
    "MSFT": "microsoft.com",
    "NVDA": "nvidia.com",
    "AMZN": "amazon.com",
    "GOOGL": "abc.xyz",
    "META": "meta.com",
    "BRK-B": "berkshirehathaway.com",
    "TSLA": "tesla.com",
    "UNH": "unitedhealthgroup.com",
    "XOM": "exxonmobil.com",
    "JNJ": "jnj.com",
    "JPM": "jpmorganchase.com",
    "V": "visa.com",
    "PG": "pg.com",
    "MA": "mastercard.com",
    "HD": "homedepot.com",
    "CVX": "chevron.com",
    "ABBV": "abbvie.com",
    "MRK": "merck.com",
    "PEP": "pepsico.com",
}


@lru_cache(maxsize=512)
def get_logo_url(ticker: str) -> str:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return ""
    domain = TICKER_DOMAIN.get(symbol)
    if not domain:
        compact = symbol.replace("-", "").replace(".", "")
        domain = f"{compact.lower()}.com"
    # Use Google's favicon endpoint for higher reliability in hosted Streamlit environments.
    return f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
