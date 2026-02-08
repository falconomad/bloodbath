import pandas as pd


TOP20 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "TSLA",
    "UNH",
    "XOM",
    "JNJ",
    "JPM",
    "V",
    "PG",
    "MA",
    "HD",
    "CVX",
    "ABBV",
    "MRK",
    "PEP",
]


def _normalize_ticker(symbol):
    """Normalize S&P 500 symbols to yfinance-compatible ticker format."""
    return str(symbol).strip().replace(".", "-")


def get_sp500_universe(fallback=None):
    """Return the latest S&P 500 constituents.

    Falls back to the provided list (or TOP20) when Wikipedia cannot be reached.
    """
    fallback_universe = fallback or TOP20

    try:
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            match="Symbol",
        )[0]
        symbols = table["Symbol"].dropna().tolist()
        normalized = [_normalize_ticker(s) for s in symbols]
        deduped = list(dict.fromkeys(normalized))
        return deduped
    except Exception:
        return fallback_universe
