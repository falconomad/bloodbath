from src.api.data_fetcher import (
    get_bulk_price_data,
    get_company_news,
    get_earnings_calendar,
    get_price_data,
    get_alpaca_snapshot_features,
)
from src.analysis.sentiment import analyze_news_sentiment
from src.analysis.technicals import calculate_technicals
from src.analysis.events import score_events
from src.analysis.dip import dip_bonus
import pandas as pd
from datetime import datetime, timezone


def _as_float(value):
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _clamp(value, min_v, max_v):
    return max(min(value, max_v), min_v)


def _trend_to_score(trend):
    if trend == "BULLISH":
        return 1.0
    if trend == "BEARISH":
        return -1.0
    return 0.0


def _safe_news(news):
    if not news:
        return []
    return [n for n in news if isinstance(n, str) and n.strip()]


def _recent_growth_score(data, lookback=20):
    if data is None or data.empty or "Close" not in data:
        return 0.0
    close = data["Close"].dropna()
    if len(close) < 2:
        return 0.0
    window = close.tail(lookback + 1)
    if len(window) < 2:
        window = close.tail(2)
    start = float(window.iloc[0])
    end = float(window.iloc[-1])
    if start <= 0:
        return 0.0
    return (end - start) / start


def _atr_percent(data, period=14):
    if data is None or data.empty:
        return 0.0
    required = {"High", "Low", "Close"}
    if not required.issubset(set(data.columns)):
        return 0.0

    frame = data[["High", "Low", "Close"]].dropna().copy()
    if len(frame) < period + 1:
        return 0.0

    prev_close = frame["Close"].shift(1)
    tr = pd.concat(
        [
            (frame["High"] - frame["Low"]).abs(),
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period).mean().iloc[-1]
    close = float(frame["Close"].iloc[-1])
    if pd.isna(atr) or close <= 0:
        return 0.0
    return float(atr) / close


def _news_quality_factor(headlines):
    n = len(headlines or [])
    if n >= 8:
        return 1.0
    if n >= 4:
        return 0.7
    if n >= 2:
        return 0.45
    return 0.2


def _technical_quality_factor(data):
    if data is None or data.empty or "Close" not in data:
        return 0.0
    close = data["Close"].dropna()
    if len(close) >= 80:
        return 1.0
    if len(close) >= 50:
        return 0.75
    if len(close) >= 30:
        return 0.45
    return 0.2


def _agreement_confidence(trend_score, sentiment, event_score):
    signs = []
    for v in [trend_score, sentiment, event_score]:
        if abs(float(v)) < 0.1:
            continue
        signs.append(1 if float(v) > 0 else -1)

    if len(signs) <= 1:
        return 0.55
    aligned = abs(sum(signs)) / len(signs)
    return 0.55 + (0.45 * aligned)


def generate_recommendation(ticker, price_data=None, news=None):
    data = price_data if price_data is not None else get_price_data(ticker)
    trend = calculate_technicals(data)
    headlines = _safe_news(news if news is not None else get_company_news(ticker))

    sentiment = float(analyze_news_sentiment(headlines)) if headlines else 0.0

    earnings = get_earnings_calendar(ticker)
    has_upcoming_earnings = len(earnings) > 0

    event_score = float(score_events(headlines, has_upcoming_earnings=has_upcoming_earnings))
    event_flag = event_score != 0.0
    micro = get_alpaca_snapshot_features(ticker)

    trend_score = _trend_to_score(trend)

    news_quality = _news_quality_factor(headlines)
    tech_quality = _technical_quality_factor(data)
    agreement_conf = _agreement_confidence(trend_score, sentiment, event_score)
    micro_quality = float(micro.get("quality", 0.0)) if micro.get("available", False) else 0.5
    signal_confidence = max(
        min((0.30 * news_quality) + (0.30 * tech_quality) + (0.25 * agreement_conf) + (0.15 * micro_quality), 1.0), 0.0
    )

    # Adaptive blend: de-emphasize news when no useful headlines are available.
    news_weight = (0.45 if headlines else 0.2) * news_quality
    event_weight = 0.20 if (headlines or has_upcoming_earnings) else 0.1
    micro_weight = 0.15 if micro.get("available", False) else 0.0
    trend_weight = max(0.1, 1.0 - news_weight - event_weight - micro_weight)

    micro_signal = (0.6 * float(micro.get("intraday_return", 0.0))) + (0.4 * (float(micro.get("rel_volume", 1.0)) - 1.0))
    micro_signal = _clamp(micro_signal * 2.0, -1.0, 1.0)
    weighted_score = (
        (trend_weight * trend_score)
        + (news_weight * sentiment)
        + (event_weight * event_score)
        + (micro_weight * micro_signal)
    )
    score = _clamp(weighted_score * 2.0, -2.0, 2.0)

    if score >= RECOMMENDATION_DECISION_THRESHOLD:
        decision = "BUY"
    elif score <= -RECOMMENDATION_DECISION_THRESHOLD:
        decision = "SELL"
    else:
        decision = "HOLD"

    confidence = _clamp(abs(score) * 50, 0, 100)

    return {
        "ticker": ticker,
        "trend": trend,
        "sentiment": sentiment,
        "signal_confidence": round(signal_confidence, 4),
        "news_quality": round(news_quality, 4),
        "technical_quality": round(tech_quality, 4),
        "micro_quality": round(micro_quality, 4),
        "upcoming_earnings": has_upcoming_earnings,
        "event_detected": event_flag,
        "event_score": event_score,
        "composite_score": round(score, 4),
        "decision": decision,
        "confidence": round(confidence, 2),
        "data": data,
    }


from src.trading.paper_trader import VirtualPortfolio

portfolio = VirtualPortfolio(starting_capital=500)


def simulate_day(ticker, decision, price):
    market = {ticker: price}

    if decision == "BUY":
        portfolio.buy(ticker, price)
    elif decision == "SELL":
        portfolio.sell(ticker, price)

    balance = portfolio.record_day(market)

    return {
        "balance": balance,
        "cash": portfolio.cash,
        "holdings": portfolio.holdings,
        "history": portfolio.balance_history,
        "log": portfolio.trade_log,
    }


from src.core.fund_manager import AutoFundManager

manager = AutoFundManager(starting_capital=500)


def run_autonomous_cycle(ticker, decision, data):
    price = _as_float(data["Close"].iloc[-1])
    manager.step(decision, price)
    return manager.get_history()


from src.core.sp500_manager import SP500AutoManager

sp500_manager = SP500AutoManager(starting_capital=500)


def run_sp500_cycle(decision, data):
    price = _as_float(data["Close"].iloc[-1])
    sp500_manager.step(decision, price)
    return sp500_manager.get_history_df(), sp500_manager.get_transactions_df()


from src.core.sp500_list import TOP20, get_sp500_universe
from src.core.top20_manager import Top20AutoManager
from src.settings import (
    TOP20_STARTING_CAPITAL,
    TOP20_MIN_BUY_SCORE,
    SIGNAL_BUY_THRESHOLD,
    SIGNAL_SELL_THRESHOLD,
    RECOMMENDATION_DECISION_THRESHOLD,
    TRADE_MODE,
    FETCH_BATCH_SIZE,
    TOP20_SLIPPAGE_BPS,
    TOP20_FEE_BPS,
)

print(
    f"[config] TRADE_MODE={TRADE_MODE} "
    f"SIGNAL_BUY_THRESHOLD={SIGNAL_BUY_THRESHOLD} SIGNAL_SELL_THRESHOLD={SIGNAL_SELL_THRESHOLD} "
    f"TOP20_MIN_BUY_SCORE={TOP20_MIN_BUY_SCORE} FETCH_BATCH_SIZE={FETCH_BATCH_SIZE} "
    f"SLIPPAGE_BPS={TOP20_SLIPPAGE_BPS} FEE_BPS={TOP20_FEE_BPS}"
)

top20_manager = Top20AutoManager(
    starting_capital=TOP20_STARTING_CAPITAL,
    min_buy_score=TOP20_MIN_BUY_SCORE,
    slippage_bps=TOP20_SLIPPAGE_BPS,
    fee_bps=TOP20_FEE_BPS,
)


def _rotation_index(num_chunks, now=None):
    current = now or datetime.now(timezone.utc)
    slot = (current.hour * 2) + (current.minute // 30)
    key = current.date().toordinal() * 48 + slot
    return key % max(num_chunks, 1)


def _rotating_fetch_universe(tickers, batch_size):
    if batch_size <= 0 or batch_size >= len(tickers):
        return tickers
    chunks = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    idx = _rotation_index(len(chunks))
    selected = chunks[idx]
    print(f"[cycle] Rotating fetch chunk {idx + 1}/{len(chunks)} (batch_size={batch_size}, tickers={len(selected)})")
    return selected


def _build_candidate_list(universe_size=120, dip_scan_size=60):
    base_universe = get_sp500_universe()
    universe_slice = base_universe[:universe_size]
    fetch_universe = _rotating_fetch_universe(universe_slice, FETCH_BATCH_SIZE)
    dip_candidates = []

    print(f"[cycle] Building candidates from {len(fetch_universe)} tickers")
    price_map = get_bulk_price_data(fetch_universe, period="6mo", interval="1d")

    for ticker in fetch_universe:
        data = price_map.get(ticker)
        if data is None:
            print(f"[cycle] {ticker}: missing bulk price data (None)")
            continue
        if data.empty:
            print(f"[cycle] {ticker}: data.empty=True in bulk price data")
            continue

        dip_score, drawdown, stabilized, volatility_penalty = dip_bonus(data)
        if drawdown is None:
            continue

        if drawdown >= 0.2:
            dip_candidates.append((ticker, data, dip_score, drawdown, stabilized, volatility_penalty))

    print(f"[cycle] Dip candidates >=20% drawdown: {len(dip_candidates)}")
    dip_candidates.sort(key=lambda item: item[3], reverse=True)
    selected_dips = dip_candidates[:dip_scan_size]

    top20_set = set(TOP20)
    final = {
        ticker: {
            "data": data,
            "dip_score": dip_score,
            "drawdown": drawdown,
            "stabilized": stabilized,
            "volatility_penalty": volatility_penalty,
        }
        for ticker, data, dip_score, drawdown, stabilized, volatility_penalty in selected_dips
    }

    for ticker in top20_set:
        if ticker not in final:
            prefetched = price_map.get(ticker)
            final[ticker] = {
                "data": prefetched if prefetched is not None else pd.DataFrame(),
                "dip_score": 0.0,
                "drawdown": None,
                "stabilized": False,
                "volatility_penalty": 0.0,
            }

    return final


def run_top20_cycle_with_signals():
    analyses = []
    candidates = _build_candidate_list()

    print(f"[cycle] Evaluating {len(candidates)} candidate tickers")
    for ticker, meta in candidates.items():
        data = meta["data"] if meta["data"] is not None else get_price_data(ticker, period="6mo", interval="1d")
        if data.empty:
            print(f"[cycle] {ticker}: skipped because data.empty=True after fetch")
            continue

        price = _as_float(data["Close"].iloc[-1])
        headlines = get_company_news(ticker)
        print(f"[cycle] {ticker}: headlines={len(headlines)}")
        rec = generate_recommendation(ticker, price_data=data, news=headlines)
        final_score = rec["composite_score"] + meta["dip_score"] + meta["volatility_penalty"]
        signal_conf = float(rec.get("signal_confidence", 0.0))

        print(
            f"[cycle] {ticker}: composite={rec['composite_score']:.4f}, "
            f"dip={meta['dip_score']:.4f}, vol_penalty={meta['volatility_penalty']:.4f}, "
            f"final={final_score:.4f}, conf={signal_conf:.3f}"
        )

        if signal_conf < 0.45:
            decision = "HOLD"
        elif final_score >= SIGNAL_BUY_THRESHOLD:
            decision = "BUY"
        elif final_score <= SIGNAL_SELL_THRESHOLD:
            decision = "SELL"
        else:
            decision = "HOLD"

        analyses.append(
            {
                "ticker": ticker,
                "decision": decision,
                "score": round(final_score, 4),
                "price": price,
                "sentiment": float(rec.get("sentiment", 0.0)),
                "growth_20d": round(_recent_growth_score(data, lookback=20), 4),
                "atr_pct": round(_atr_percent(data, period=14), 5),
                "signal_confidence": round(signal_conf, 4),
            }
        )

    # Always mark-to-market current holdings, even if they were not in the current candidate slice.
    held_tickers = set(top20_manager.holdings.keys())
    analyzed_tickers = {a["ticker"] for a in analyses}
    missing_held = sorted(held_tickers - analyzed_tickers)
    for ticker in missing_held:
        data = get_price_data(ticker, period="6mo", interval="1d")
        if data.empty:
            print(f"[cycle] {ticker}: mark-to-market skipped (data.empty=True)")
            continue
        price = _as_float(data["Close"].iloc[-1])
        analyses.append(
            {
                "ticker": ticker,
                "decision": "HOLD",
                "score": 0.0,
                "price": price,
                "sentiment": 0.0,
                "growth_20d": round(_recent_growth_score(data, lookback=20), 4),
                "atr_pct": round(_atr_percent(data, period=14), 5),
                "signal_confidence": 0.5,
            }
        )
        print(f"[cycle] {ticker}: mark-to-market price={price:.2f}")

    buys = sum(1 for a in analyses if a["decision"] == "BUY")
    sells = sum(1 for a in analyses if a["decision"] == "SELL")
    holds = sum(1 for a in analyses if a["decision"] == "HOLD")
    print(f"[cycle] Signals => BUY:{buys} SELL:{sells} HOLD:{holds}")
    top20_manager.step(analyses)

    positions = top20_manager.position_snapshot_df()
    return top20_manager.history_df(), top20_manager.transactions_df(), positions, analyses


def run_top20_cycle():
    history, transactions, positions, _analyses = run_top20_cycle_with_signals()
    return history, transactions, positions
