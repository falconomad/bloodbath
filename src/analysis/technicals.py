import pandas as pd


def _latest_scalar(value):
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


def calculate_technicals(df):
    close = df["Close"]
    df["SMA20"] = close.rolling(window=20).mean()
    df["SMA50"] = close.rolling(window=50).mean()

    latest = df.iloc[-1]
    sma20 = _latest_scalar(latest["SMA20"])
    sma50 = _latest_scalar(latest["SMA50"])

    trend = "BULLISH" if sma20 > sma50 else "BEARISH"

    return trend
