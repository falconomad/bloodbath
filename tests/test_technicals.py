import pandas as pd

from src.analysis.technicals import calculate_technicals


def test_returns_neutral_for_short_history():
    df = pd.DataFrame({"Close": [100 + i for i in range(20)]})
    assert calculate_technicals(df) == "NEUTRAL"


def test_returns_neutral_for_missing_data():
    df = pd.DataFrame({"Close": [None] * 60})
    assert calculate_technicals(df) == "NEUTRAL"


def test_returns_bullish_for_uptrend():
    df = pd.DataFrame({"Close": [100 + i for i in range(80)]})
    assert calculate_technicals(df) == "BULLISH"


def test_handles_duplicate_close_columns_without_ambiguous_series_error():
    close = [100 + i for i in range(80)]
    df = pd.DataFrame(list(zip(close, close)), columns=["Close", "Close"])
    assert calculate_technicals(df) == "BULLISH"
