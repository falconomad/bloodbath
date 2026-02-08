from src.api.data_fetcher import get_bulk_price_data, get_company_news, get_earnings_calendar, get_price_data
from src.analysis.sentiment import analyze_news_sentiment
from src.analysis.technicals import calculate_technicals
from src.analysis.events import score_events
from src.analysis.dip import dip_bonus


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


def generate_recommendation(ticker, price_data=None, news=None):
    data = price_data if price_data is not None else get_price_data(ticker)
    trend = calculate_technicals(data)
    headlines = _safe_news(news if news is not None else get_company_news(ticker))

    sentiment = float(analyze_news_sentiment(headlines)) if headlines else 0.0

    earnings = get_earnings_calendar(ticker)
    has_upcoming_earnings = len(earnings) > 0

    event_score = float(score_events(headlines, has_upcoming_earnings=has_upcoming_earnings))
    event_flag = event_score != 0.0

    trend_score = _trend_to_score(trend)

    # Adaptive blend: de-emphasize news when no useful headlines are available.
    news_weight = 0.45 if headlines else 0.2
    event_weight = 0.25 if (headlines or has_upcoming_earnings) else 0.1
    trend_weight = 1.0 - news_weight - event_weight

    weighted_score = (trend_weight * trend_score) + (news_weight * sentiment) + (event_weight * event_score)
    score = _clamp(weighted_score * 2.0, -2.0, 2.0)

    if score >= 0.7:
        decision = "BUY"
    elif score <= -0.7:
        decision = "SELL"
    else:
        decision = "HOLD"

    confidence = _clamp(abs(score) * 50, 0, 100)

    return {
        "ticker": ticker,
        "trend": trend,
        "sentiment": sentiment,
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

top20_manager = Top20AutoManager(starting_capital=500)


def _build_candidate_list(universe_size=120, dip_scan_size=60):
    base_universe = get_sp500_universe()
    universe_slice = base_universe[:universe_size]
    dip_candidates = []

    price_map = get_bulk_price_data(universe_slice, period="6mo", interval="1d")

    for ticker in universe_slice:
        data = price_map.get(ticker)
        if data is None or data.empty:
            continue

        dip_score, drawdown, stabilized, volatility_penalty = dip_bonus(data)
        if drawdown is None:
            continue

        if drawdown >= 0.2:
            dip_candidates.append((ticker, data, dip_score, drawdown, stabilized, volatility_penalty))

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
            final[ticker] = {
                "data": None,
                "dip_score": 0.0,
                "drawdown": None,
                "stabilized": False,
                "volatility_penalty": 0.0,
            }

    return final


def run_top20_cycle():
    analyses = []
    candidates = _build_candidate_list()

    for ticker, meta in candidates.items():
        data = meta["data"] if meta["data"] is not None else get_price_data(ticker, period="6mo", interval="1d")
        if data.empty:
            continue

        price = _as_float(data["Close"].iloc[-1])
        headlines = get_company_news(ticker)
        rec = generate_recommendation(ticker, price_data=data, news=headlines)
        final_score = rec["composite_score"] + meta["dip_score"] + meta["volatility_penalty"]

        if final_score > 1:
            decision = "BUY"
        elif final_score < -1:
            decision = "SELL"
        else:
            decision = "HOLD"

        analyses.append(
            {
                "ticker": ticker,
                "decision": decision,
                "score": round(final_score, 4),
                "price": price,
            }
        )

    if analyses:
        top20_manager.step(analyses)

    positions = top20_manager.position_snapshot_df()
    return top20_manager.history_df(), top20_manager.transactions_df(), positions
