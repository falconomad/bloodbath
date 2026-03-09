from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from datetime import datetime, timedelta, timezone
import logging
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

import config

news_client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET)

def get_news_context(symbols, limit_per_symbol=3):
    """Retrieve recent news headlines and summaries for given symbols."""
    news_by_symbol = {str(s).upper(): [] for s in symbols}
    try:
        req = NewsRequest(symbols=",".join(symbols), limit=min(50, len(symbols)*limit_per_symbol))
        res = news_client.get_news(req)
        
        # Depending on alpaca-py version, news might be in res.data["news"] or res.news
        articles = res.data.get("news", []) if hasattr(res, 'data') and isinstance(res.data, dict) else getattr(res, 'news', [])
        if not articles and hasattr(res, 'news'):
            articles = res.news
            
        for article in articles:
            for sym in article.symbols:
                sym_u = str(sym).upper()
                if sym_u not in news_by_symbol:
                    news_by_symbol[sym_u] = []
                if len(news_by_symbol[sym_u]) < limit_per_symbol:
                     news_by_symbol[sym_u].append({
                         "headline": article.headline,
                         "summary": getattr(article, 'summary', ''),
                         "date": article.created_at.isoformat(),
                         "source": "alpaca",
                     })
    except Exception as e:
        logging.error(f"[NewsFeed] Error fetching news: {e}", exc_info=True)
    # Fallback/fill via yfinance news per symbol when available.
    if yf is None:
        return news_by_symbol
    now = datetime.now(timezone.utc)
    min_time = now - timedelta(days=3)
    for sym in symbols:
        sym_u = str(sym).upper()
        existing = news_by_symbol.get(sym_u, [])
        if len(existing) >= limit_per_symbol:
            continue
        try:
            ticker = yf.Ticker(sym_u)
            items = ticker.news or []
            for item in items:
                if len(existing) >= limit_per_symbol:
                    break
                pub_ts = item.get("providerPublishTime")
                if pub_ts:
                    dt = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
                    if dt < min_time:
                        continue
                    dts = dt.isoformat()
                else:
                    dts = now.isoformat()
                existing.append(
                    {
                        "headline": item.get("title", ""),
                        "summary": item.get("summary", "") or "",
                        "date": dts,
                        "source": str(item.get("publisher", "yfinance")).strip().lower() or "yfinance",
                    }
                )
            news_by_symbol[sym_u] = existing[:limit_per_symbol]
        except Exception:
            continue
    return news_by_symbol
