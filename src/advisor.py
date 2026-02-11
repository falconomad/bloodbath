from src.api.data_fetcher import (
    get_bulk_price_data,
    get_company_news,
    get_market_sentiment_news,
    get_price_data,
    reset_cycle_caches,
)
from src.analysis.dip import dip_bonus
from src.pipeline.decision_engine import (
    load_config,
)
from src.pipeline.recommendation_service import (
    atr_percent,
    generate_recommendation_core,
)
import pandas as pd
from datetime import datetime, timezone


def _as_float(value):
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


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


SCORING_CONFIG = load_config()
_DECISION_STATE = {}
_CYCLE_INDEX = 0


def export_decision_state():
    return {str(k): dict(v) for k, v in _DECISION_STATE.items()}


def import_decision_state(state):
    global _DECISION_STATE
    if not isinstance(state, dict):
        _DECISION_STATE = {}
        return
    parsed = {}
    for ticker, payload in state.items():
        if not isinstance(payload, dict):
            continue
        parsed[str(ticker)] = {
            "decision": str(payload.get("decision", "HOLD")),
            "cycle": int(payload.get("cycle", 0)),
            "flip_cycle": int(payload.get("flip_cycle", 0)),
            "last_non_hold_cycle": int(payload.get("last_non_hold_cycle", -10_000)),
        }
    _DECISION_STATE = parsed


def get_cycle_index():
    return int(_CYCLE_INDEX)


def set_cycle_index(value):
    global _CYCLE_INDEX
    try:
        _CYCLE_INDEX = int(value)
    except Exception:
        _CYCLE_INDEX = 0


def generate_recommendation(
    ticker,
    price_data=None,
    news=None,
    dip_meta=None,
    cycle_idx=0,
    apply_stability_gate=False,
    portfolio_context=None,
    market_news=None,
):
    data = price_data if price_data is not None else get_price_data(ticker)
    headlines = news if news is not None else get_company_news(ticker, structured=True)
    return generate_recommendation_core(
        ticker=ticker,
        data=data,
        headlines=headlines,
        dip_meta=dip_meta,
        cycle_idx=cycle_idx,
        apply_stability_gate=apply_stability_gate,
        cfg=SCORING_CONFIG,
        decision_state=_DECISION_STATE,
        portfolio_context=portfolio_context,
        market_news=market_news,
    )


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


from src.core.sp500_list import TOP20_SECTOR, get_sp500_universe
from src.core.top20_manager import Top20AutoManager
from src.settings import (
    ENABLE_POSITION_ROTATION,
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
    enable_position_rotation=ENABLE_POSITION_ROTATION,
    rotation_min_score_gap=0.15,
    rotation_sell_fraction=0.35,
    rotation_max_swaps_per_step=1,
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


def _latest_daily_return(data):
    if data is None or data.empty or "Close" not in data:
        return None
    close = data["Close"].dropna()
    if len(close) < 2:
        return None
    prev_close = float(close.iloc[-2])
    last_close = float(close.iloc[-1])
    if prev_close <= 0:
        return None
    return (last_close - prev_close) / prev_close


def _build_candidate_list(universe_size=500, top_losers=10, top_gainers=10):
    base_universe = get_sp500_universe()
    universe_slice = base_universe[:universe_size]
    fetch_universe = _rotating_fetch_universe(universe_slice, FETCH_BATCH_SIZE) if FETCH_BATCH_SIZE >= universe_size else universe_slice

    print(f"[cycle] Building candidates from {len(fetch_universe)} tickers")
    price_map = get_bulk_price_data(fetch_universe, period="6mo", interval="1d")
    movers = []
    for ticker, data in price_map.items():
        if data is None or data.empty:
            continue
        daily_ret = _latest_daily_return(data)
        if daily_ret is None:
            continue
        movers.append((ticker, data, daily_ret))

    losers = sorted(movers, key=lambda item: item[2])[: max(int(top_losers), 0)]
    gainers = sorted(movers, key=lambda item: item[2], reverse=True)[: max(int(top_gainers), 0)]
    selected = losers + [item for item in gainers if item[0] not in {row[0] for row in losers}]

    print(
        f"[cycle] Movers selected: losers={len(losers)} gainers={len(gainers)} "
        f"from_ranked={len(movers)}"
    )
    final = {}
    for ticker, data, daily_ret in selected:
        dip_score, drawdown, stabilized, volatility_penalty = dip_bonus(data)
        final[ticker] = {
            "data": data,
            "dip_score": dip_score,
            "drawdown": drawdown,
            "stabilized": stabilized,
            "volatility_penalty": volatility_penalty,
            "daily_return": daily_ret,
            "mover_bucket": "LOSER" if daily_ret < 0 else "GAINER",
        }

    return final


def _portfolio_risk_context_for_ticker(ticker, analyses_so_far, candidates):
    current_value = float(top20_manager.portfolio_value())
    history = [float(v) for v in top20_manager.history] + [current_value]
    rolling_peak = max(history) if history else max(current_value, 1.0)
    portfolio_drawdown = max((rolling_peak - current_value) / rolling_peak, 0.0) if rolling_peak > 0 else 0.0

    sector_mv = {}
    total_mv = max(current_value, 1e-9)
    for held_ticker, position in top20_manager.holdings.items():
        price = float(top20_manager.last_price_by_ticker.get(held_ticker, position.get("avg_cost", 0.0)))
        shares = float(position.get("shares", 0.0))
        mv = max(shares * max(price, 0.0), 0.0)
        sector = str(position.get("sector", TOP20_SECTOR.get(held_ticker, "UNKNOWN")))
        sector_mv[sector] = sector_mv.get(sector, 0.0) + mv

    # Include currently proposed buys in running sector exposure estimate.
    for rec in analyses_so_far:
        if str(rec.get("decision", "")).upper() != "BUY":
            continue
        sec = str(rec.get("sector", TOP20_SECTOR.get(rec.get("ticker", ""), "UNKNOWN")))
        alloc_hint = float(rec.get("position_size", 0.0))
        sector_mv[sec] = sector_mv.get(sec, 0.0) + (alloc_hint * current_value)

    ticker_sector = TOP20_SECTOR.get(ticker, "UNKNOWN")
    ticker_sector_allocation = float(sector_mv.get(ticker_sector, 0.0)) / total_mv if total_mv > 0 else 0.0

    held_or_candidate = list({*top20_manager.holdings.keys(), ticker})
    returns = {}
    for sym in held_or_candidate:
        data = (candidates.get(sym) or {}).get("data")
        if data is None or data.empty or "Close" not in data:
            continue
        close = data["Close"].dropna()
        if len(close) < 25:
            continue
        returns[sym] = close.pct_change().dropna().tail(40)
    corr_values = []
    if len(returns) >= 2:
        aligned = pd.DataFrame(returns).dropna(how="any")
        if aligned.shape[0] >= 10:
            corr = aligned.corr().abs()
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i + 1 :]:
                    val = float(corr.loc[c1, c2])
                    if pd.notna(val):
                        corr_values.append(val)
    portfolio_avg_correlation = float(sum(corr_values) / len(corr_values)) if corr_values else 0.0

    return {
        "portfolio_drawdown": round(portfolio_drawdown, 6),
        "portfolio_avg_correlation": round(portfolio_avg_correlation, 6),
        "ticker_sector_allocation": round(ticker_sector_allocation, 6),
    }


def run_top20_cycle_with_signals():
    global _CYCLE_INDEX
    _CYCLE_INDEX += 1
    reset_cycle_caches()
    analyses = []
    candidates = _build_candidate_list()
    cycle_market_news = get_market_sentiment_news(limit=12)

    print(f"[cycle] Evaluating {len(candidates)} candidate tickers")
    for ticker, meta in candidates.items():
        data = meta["data"] if meta["data"] is not None else get_price_data(ticker, period="6mo", interval="1d")
        if data.empty:
            print(f"[cycle] {ticker}: skipped because data.empty=True after fetch")
            continue

        price = _as_float(data["Close"].iloc[-1])
        headlines = get_company_news(ticker, structured=True)
        print(f"[cycle] {ticker}: headlines={len(headlines)}")
        rec = generate_recommendation(
            ticker,
            price_data=data,
            news=headlines,
            dip_meta=meta,
            cycle_idx=_CYCLE_INDEX,
            apply_stability_gate=True,
            portfolio_context=_portfolio_risk_context_for_ticker(ticker, analyses, candidates),
            market_news=cycle_market_news,
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
                "sector": TOP20_SECTOR.get(ticker, "UNKNOWN"),
                "sentiment": float(rec.get("sentiment", 0.0)),
                "social_post_count": int(rec.get("social_post_count", 0)),
                "market_news_count": int(rec.get("market_news_count", 0)),
                "social_quality": float(rec.get("social_quality", 0.0)),
                "growth_20d": round(_recent_growth_score(data, lookback=20), 4),
                "atr_pct": round(atr_percent(data, period=14), 5),
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
                "sector": TOP20_SECTOR.get(ticker, "UNKNOWN"),
                "sentiment": 0.0,
                "growth_20d": round(_recent_growth_score(data, lookback=20), 4),
                "atr_pct": round(atr_percent(data, period=14), 5),
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
