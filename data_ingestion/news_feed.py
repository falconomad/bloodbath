from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

import config

news_client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET)

def get_news_context(symbols, limit_per_symbol=3):
    """Retrieve recent news headlines and summaries for given symbols."""
    try:
        req = NewsRequest(symbols=",".join(symbols), limit=min(50, len(symbols)*limit_per_symbol))
        res = news_client.get_news(req)
        
        # Depending on alpaca-py version, news might be in res.data["news"] or res.news
        articles = res.data.get("news", []) if hasattr(res, 'data') and isinstance(res.data, dict) else getattr(res, 'news', [])
        if not articles and hasattr(res, 'news'):
            articles = res.news
            
        news_by_symbol = {}
        for article in articles:
            for sym in article.symbols:
                if sym not in news_by_symbol:
                    news_by_symbol[sym] = []
                if len(news_by_symbol[sym]) < limit_per_symbol:
                     news_by_symbol[sym].append({
                         "headline": article.headline,
                         "summary": getattr(article, 'summary', ''),
                         "date": article.created_at.isoformat()
                     })
        return news_by_symbol
    except Exception as e:
        print(f"[NewsFeed] Error fetching news: {e}")
        return {}
