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

    # Deterministic technical score: momentum + relative strength + volume confirmation.
    score = 50.0
    score += max(min(asset_ret * 6.0, 30.0), -30.0)
    score += max(min(rel_strength * 4.0, 20.0), -20.0)
    score += max(min((vol_spike - 1.0) * 15.0, 12.0), -8.0)
    score = max(0, min(100, int(round(score))))

    rationale = (
        f"5-day return {asset_ret:.2f}%, relative strength vs SPY {rel_strength:.2f}%, "
        f"latest volume multiple {vol_spike:.2f}x."
    )
    return {"score": score, "rationale": rationale}
