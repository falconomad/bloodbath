import pandas as pd


def _latest_scalar(value):
    if isinstance(value, pd.Series):
        value = value.dropna()
        if value.empty:
            return float("nan")
        return float(value.iloc[-1])
    return float(value)


def calculate_technicals(df):
    if df is None or "Close" not in df or df.empty:
        return "NEUTRAL"

    close = df["Close"].dropna()
    if len(close) < 50:
        return "NEUTRAL"

    sma20 = close.rolling(window=20).mean().iloc[-1]
    sma50 = close.rolling(window=50).mean().iloc[-1]

    sma20 = _latest_scalar(sma20)
    sma50 = _latest_scalar(sma50)

    if pd.isna(sma20) or pd.isna(sma50):
        return "NEUTRAL"

    trend = "BULLISH" if sma20 > sma50 else "BEARISH"

    return trend
