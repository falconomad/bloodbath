from src.api.data_fetcher import (
    get_bulk_price_data,
    get_company_news,
    get_price_data,
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


def generate_recommendation(ticker, price_data=None, news=None, dip_meta=None, cycle_idx=0, apply_stability_gate=False):
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


from src.core.sp500_list import TOP20, TOP20_SECTOR, get_sp500_universe
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
        headlines = get_company_news(ticker, structured=True)
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
                "sector": TOP20_SECTOR.get(ticker, "UNKNOWN"),
                "sentiment": float(rec.get("sentiment", 0.0)),
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
