
def calculate_technicals(df):

    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    latest = df.iloc[-1]

    trend = "BULLISH" if latest["SMA20"] > latest["SMA50"] else "BEARISH"

    return trend
