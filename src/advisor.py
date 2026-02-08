from src.api.data_fetcher import get_price_data, get_company_news
from src.analysis.sentiment import analyze_news_sentiment
from src.analysis.technicals import calculate_technicals
from src.analysis.events import score_events


def _as_float(value):
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def generate_recommendation(ticker, price_data=None, news=None):
    data = price_data if price_data is not None else get_price_data(ticker)
    trend = calculate_technicals(data)
    headlines = news if news is not None else get_company_news(ticker)
    sentiment = analyze_news_sentiment(headlines)
    event_score = score_events(headlines)
    event_flag = event_score != 0.0

    score = 0

    if trend == "BULLISH":
        score += 1
    else:
        score -= 1

    score += sentiment

    score += event_score

    if score > 1:
        decision = "BUY"
    elif score < -1:
        decision = "SELL"
    else:
        decision = "HOLD"

    confidence = min(abs(score) * 20, 100)

    return {
        "ticker": ticker,
        "trend": trend,
        "sentiment": sentiment,
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


from src.core.sp500_list import TOP20
from src.core.top20_manager import Top20AutoManager

top20_manager = Top20AutoManager(starting_capital=500)


def run_top20_cycle():
    analyses = []

    for ticker in TOP20:
        data = get_price_data(ticker, period="6mo", interval="1d")
        if data.empty:
            continue

        price = _as_float(data["Close"].iloc[-1])
        headlines = get_company_news(ticker)
        rec = generate_recommendation(ticker, price_data=data, news=headlines)

        analyses.append(
            {
                "ticker": ticker,
                "decision": rec["decision"],
                "score": rec["composite_score"],
                "price": price,
            }
        )

    if analyses:
        top20_manager.step(analyses)

    return top20_manager.history_df(), top20_manager.transactions_df()
