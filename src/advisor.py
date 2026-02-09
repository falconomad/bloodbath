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
from src.pipeline.decision_engine import (
    Signal,
    aggregate_confidence,
    clamp,
    decide,
    load_config,
    normalize_signals,
    weighted_score,
    write_trace,
)
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


def _price_data_quality_ok(data, min_points=40, max_missing_ratio=0.05):
    if data is None or data.empty or "Close" not in data:
        return False
    close = data["Close"]
    if len(close) < int(min_points):
        return False
    missing_ratio = float(close.isna().mean())
    return missing_ratio <= float(max_missing_ratio)


def _price_data_gap_ratio(data):
    if data is None or data.empty or "Close" not in data:
        return 1.0
    close = data["Close"]
    return float(close.isna().mean())


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


SCORING_CONFIG = load_config()
_DECISION_STATE = {}
_CYCLE_INDEX = 0


def _normalized_module_signals(ticker, data, headlines, dip_meta=None):
    dip_meta = dip_meta or {}
    quality_cfg = SCORING_CONFIG.get("quality", {})
    risk_cfg = SCORING_CONFIG.get("risk", {})

    trend = calculate_technicals(data)
    trend_score = _trend_to_score(trend)
    tech_quality = _technical_quality_factor(data)
    price_ok = _price_data_quality_ok(
        data,
        min_points=quality_cfg.get("min_price_points", 40),
        max_missing_ratio=quality_cfg.get("max_missing_ratio", 0.05),
    )
    data_gap_ratio = _price_data_gap_ratio(data)
    trend_signal = Signal(
        name="trend",
        value=trend_score,
        confidence=(0.2 + (0.8 * tech_quality)) if price_ok else 0.0,
        quality_ok=price_ok,
        reason="" if price_ok else "insufficient_price_history_or_gaps",
    )

    sentiment = float(analyze_news_sentiment(headlines)) if headlines else 0.0
    min_sentiment_articles = int(quality_cfg.get("min_sentiment_articles", 4))
    sentiment_ok = len(headlines) >= min_sentiment_articles
    sentiment_variance = min(abs(sentiment), 1.0)
    sentiment_signal = Signal(
        name="sentiment",
        value=sentiment,
        confidence=(0.35 + 0.65 * sentiment_variance) if sentiment_ok else 0.0,
        quality_ok=sentiment_ok,
        reason="" if sentiment_ok else "too_few_articles",
    )

    earnings = get_earnings_calendar(ticker)
    try:
        has_upcoming_earnings = len(earnings) > 0
    except Exception:
        has_upcoming_earnings = False
    event_score = float(score_events(headlines, has_upcoming_earnings=has_upcoming_earnings))
    min_news_articles = int(quality_cfg.get("min_news_articles", 3))
    events_ok = (len(headlines) >= min_news_articles) or has_upcoming_earnings
    event_signal = Signal(
        name="events",
        value=event_score,
        confidence=0.6 if events_ok else 0.0,
        quality_ok=events_ok,
        reason="" if events_ok else "insufficient_event_context",
    )

    micro = get_alpaca_snapshot_features(ticker)
    micro_available = isinstance(micro, dict) and bool(micro.get("available", False))
    rel_volume = float(micro.get("rel_volume", 1.0)) if micro_available else 1.0
    intraday_return = float(micro.get("intraday_return", 0.0)) if micro_available else 0.0
    micro_quality = float(micro.get("quality", 0.0)) if micro_available else 0.0
    micro_signal_value = clamp((0.6 * intraday_return) + (0.4 * (rel_volume - 1.0)), -1.0, 1.0)
    micro_ok = micro_available and micro_quality > 0.0
    micro_signal = Signal(
        name="micro",
        value=micro_signal_value,
        confidence=micro_quality if micro_ok else 0.0,
        quality_ok=micro_ok,
        reason="" if micro_ok else "micro_data_unavailable",
    )

    dip_score = float(dip_meta.get("dip_score", 0.0))
    dip_signal = Signal(
        name="dip",
        value=clamp(dip_score, -1.0, 1.0),
        confidence=0.6 if dip_meta.get("drawdown") is not None else 0.0,
        quality_ok=dip_meta.get("drawdown") is not None,
        reason="" if dip_meta.get("drawdown") is not None else "no_drawdown_context",
    )
    vol_penalty = float(dip_meta.get("volatility_penalty", 0.0))
    vol_scale = max(float(risk_cfg.get("max_atr_pct_for_full_risk", 0.08)), 1e-6)
    volatility_signal = Signal(
        name="volatility",
        value=clamp(vol_penalty / vol_scale, -1.0, 0.0),
        confidence=0.7 if dip_meta.get("drawdown") is not None else 0.0,
        quality_ok=dip_meta.get("drawdown") is not None,
        reason="" if dip_meta.get("drawdown") is not None else "no_volatility_context",
    )

    signals = normalize_signals(
        {
            "trend": trend_signal,
            "sentiment": sentiment_signal,
            "events": event_signal,
            "micro": micro_signal,
            "dip": dip_signal,
            "volatility": volatility_signal,
        }
    )
    return trend, has_upcoming_earnings, signals, {
        "rel_volume": rel_volume,
        "micro_available": micro_available,
        "data_quality_ok": price_ok,
        "data_gap_ratio": data_gap_ratio,
        "atr_pct": _atr_percent(data, period=14),
        "article_count": len(headlines),
    }


def generate_recommendation(ticker, price_data=None, news=None, dip_meta=None, cycle_idx=0, apply_stability_gate=False):
    data = price_data if price_data is not None else get_price_data(ticker)
    headlines = _safe_news(news if news is not None else get_company_news(ticker))
    trend, has_upcoming_earnings, signals, risk_context = _normalized_module_signals(
        ticker=ticker, data=data, headlines=headlines, dip_meta=dip_meta
    )
    weights = SCORING_CONFIG.get("weights", {})
    score = weighted_score(signals, weights)
    signal_confidence, conflicts = aggregate_confidence(
        signals,
        SCORING_CONFIG.get("risk", {}).get("conflict_penalty", 0.2),
        weights=weights,
        max_conflict_drop=SCORING_CONFIG.get("risk", {}).get("max_confidence_drop_on_conflict", 0.75),
    )
    signal_confidence = clamp(signal_confidence, 0.0, 1.0)
    event_score = float(signals["events"].value)
    sentiment = float(signals["sentiment"].value)
    event_flag = abs(event_score) > 0.0
    decision, decision_reasons, suggested_position_size = decide(
        ticker=ticker,
        score=score,
        confidence=signal_confidence,
        signals=signals,
        risk_context=risk_context,
        state=_DECISION_STATE if apply_stability_gate else {},
        cycle_idx=cycle_idx,
        cfg=SCORING_CONFIG,
    )
    confidence = signal_confidence * 100.0

    write_trace(
        {
            "ticker": ticker,
            "score": round(score, 6),
            "confidence": round(signal_confidence, 6),
            "decision": decision,
            "decision_reasons": decision_reasons,
            "weights": weights,
            "signals": {k: s.as_dict() for k, s in signals.items()},
            "conflict_ratio": round(conflicts, 6),
            "risk_context": risk_context,
        }
    )

    return {
        "ticker": ticker,
        "trend": trend,
        "sentiment": sentiment,
        "signal_confidence": round(signal_confidence, 4),
        "news_quality": round(_news_quality_factor(headlines), 4),
        "technical_quality": round(_technical_quality_factor(data), 4),
        "micro_quality": round(signals["micro"].confidence, 4),
        "upcoming_earnings": has_upcoming_earnings,
        "event_detected": event_flag,
        "event_score": event_score,
        "composite_score": round(score, 4),
        "decision": decision,
        "confidence": round(confidence, 2),
        "decision_reasons": decision_reasons,
        "position_size": suggested_position_size,
        "signals": {k: s.as_dict() for k, s in signals.items()},
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
    TRADE_MODE,
    FETCH_BATCH_SIZE,
    TOP20_SLIPPAGE_BPS,
    TOP20_FEE_BPS,
)

print(
    f"[config] TRADE_MODE={TRADE_MODE} "
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
    global _CYCLE_INDEX
    _CYCLE_INDEX += 1
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
        rec = generate_recommendation(
            ticker,
            price_data=data,
            news=headlines,
            dip_meta=meta,
            cycle_idx=_CYCLE_INDEX,
            apply_stability_gate=True,
        )
        final_score = float(rec["composite_score"])
        signal_conf = float(rec.get("signal_confidence", 0.0))

        print(
            f"[cycle] {ticker}: final={final_score:.4f}, conf={signal_conf:.3f}, "
            f"decision={rec.get('decision', 'HOLD')}"
        )
        decision = rec.get("decision", "HOLD")

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
                "position_size": float(rec.get("position_size", 0.0)),
                "decision_reasons": rec.get("decision_reasons", []),
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
