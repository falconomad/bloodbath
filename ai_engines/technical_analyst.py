import json
from statistics import mean

def evaluate_technicals(symbol, history, spy_context):
    """
    Grades a stock strictly on technical trends (price action, momentum)
    Returns a score from 0-100 and a 1-sentence rationale.
    """
    if not history or len(history) < 2:
        return {"score": 0, "rationale": "Insufficient price history for technical analysis."}

    closes = [float(x.get("c", 0.0)) for x in history if float(x.get("c", 0.0)) > 0]
    vols = [float(x.get("v", 0.0)) for x in history if float(x.get("v", 0.0)) >= 0]
    if len(closes) < 2:
        return {"score": 0, "rationale": "Insufficient close-price points."}

    asset_ret = ((closes[-1] - closes[0]) / closes[0]) * 100.0

    spy_ret = 0.0
    spy_closes = [float(x.get("c", 0.0)) for x in (spy_context or []) if float(x.get("c", 0.0)) > 0]
    if len(spy_closes) >= 2:
        spy_ret = ((spy_closes[-1] - spy_closes[0]) / spy_closes[0]) * 100.0

    rel_strength = asset_ret - spy_ret
    avg_vol = mean(vols) if vols else 0.0
    vol_spike = (vols[-1] / avg_vol) if avg_vol > 0 else 1.0

    # 1. SMA Trend confirmation (Short-term 3-day Simple Moving Average)
    sma_bonus = 0.0
    if len(closes) >= 3:
        sma3 = mean(closes[-3:])
        if closes[-1] > sma3:
            sma_bonus = 15.0
        else:
            sma_bonus = -15.0

    # 2. Volume trend acceleration
    vol_acceleration_bonus = 0.0
    if len(vols) >= 3:
        if vols[-1] > vols[-2] and vols[-2] > vols[-3]:
            vol_acceleration_bonus = 10.0
        elif vols[-1] > vols[-2]:
            vol_acceleration_bonus = 5.0
        else:
            vol_acceleration_bonus = -5.0

    # 3. Price acceleration (momentum rate of change)
    price_acceleration = 0.0
    if len(closes) >= 3:
        day1_ret = ((closes[-1] - closes[-2]) / closes[-2]) * 100.0
        day2_ret = ((closes[-2] - closes[-3]) / closes[-3]) * 100.0
        price_acceleration = day1_ret - day2_ret

    # Deterministic technical score calculation
    score = 50.0
    
    # Trend (SMA) component: up to ±15
    score += sma_bonus
    
    # Momentum (return + price acceleration) component: up to ±25
    momentum_factor = (asset_ret * 4.0) + (price_acceleration * 3.0)
    score += max(min(momentum_factor, 25.0), -25.0)
    
    # Relative strength component: up to ±20
    score += max(min(rel_strength * 4.0, 20.0), -20.0)
    
    # Volume spike & acceleration component: up to ±20
    vol_factor = ((vol_spike - 1.0) * 10.0) + vol_acceleration_bonus
    score += max(min(vol_factor, 20.0), -20.0)
    
    score = max(0, min(100, int(round(score))))

    rationale = (
        f"5-day return {asset_ret:.2f}%, relative strength vs SPY {rel_strength:.2f}%, "
        f"SMA3 position {'above' if sma_bonus > 0 else 'below'}, volume spike {vol_spike:.2f}x."
    )
    return {"score": score, "rationale": rationale}
