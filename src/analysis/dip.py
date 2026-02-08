import pandas as pd


def recent_drawdown_ratio(data, lookback_days=120):
    closes = data["Close"].dropna()
    if closes.empty:
        return None

    lookback_window = closes.tail(lookback_days)
    if lookback_window.empty:
        return None

    peak = float(lookback_window.max())
    current = float(lookback_window.iloc[-1])
    if peak <= 0:
        return None

    return (peak - current) / peak


def stabilization_signal(data, window=5):
    """Lightweight falling-knife check.

    Returns True when the latest close is stabilizing around short-term mean and
    is not in a steep week-over-week decline.
    """
    closes = data["Close"].dropna()
    if len(closes) < window + 1:
        return False

    recent = closes.tail(window)
    latest = float(recent.iloc[-1])
    short_sma = float(recent.mean())
    week_change = (latest - float(closes.iloc[-window - 1])) / max(float(closes.iloc[-window - 1]), 1e-9)

    return latest >= short_sma and week_change >= -0.02


def volatility_risk_penalty(data, window=20):
    """Return a small negative score for highly volatile names.

    Uses annualized realized volatility from daily returns.
    """
    closes = data["Close"].dropna()
    if len(closes) < window + 1:
        return 0.0

    returns = closes.pct_change().dropna().tail(window)
    if returns.empty:
        return 0.0

    daily_vol = float(returns.std())
    annualized_vol = daily_vol * (252 ** 0.5)

    if annualized_vol < 0.45:
        return 0.0
    if annualized_vol < 0.60:
        return -0.1
    if annualized_vol < 0.80:
        return -0.25
    return -0.4


def dip_bonus(data, dip_threshold=0.2, lookback_days=120):
    drawdown = recent_drawdown_ratio(data, lookback_days=lookback_days)
    if drawdown is None:
        return 0.0, None, False, 0.0

    penalty = volatility_risk_penalty(data)

    if drawdown < dip_threshold:
        return 0.0, drawdown, False, penalty

    stabilized = stabilization_signal(data)

    # Cap extra boost so temporary panic dips are considered,
    # but never dominate sentiment/technical signals.
    dip_score = min((drawdown - dip_threshold) * 4, 1.5)
    if not stabilized:
        dip_score *= 0.5

    return round(dip_score, 4), drawdown, stabilized, penalty
