from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest
from alpaca.data.models.screener import MarketType
from datetime import datetime, timedelta, timezone
import logging

import config

data_client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET)
screener_client = ScreenerClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET)

def get_top_movers(limit=15):
    """Dynamically scan the market for the top gainers and losers of the day."""
    try:
        req = MarketMoversRequest(
            top=limit,
            market_type=MarketType.STOCKS
        )
        response = screener_client.get_market_movers(req)
        
        movers = []
        for mover in response.gainers + response.losers:
            movers.append({
                "symbol": mover.symbol,
                "price": float(mover.price),
                "change_pct": float(mover.percent_change)
            })
            
        return movers
    except Exception as e:
        logging.error(f"[MarketData] Error fetching top movers via Screener API: {e}", exc_info=True)
        return []

def get_history_context(symbols, days_back=5):
    """Fetch recent price history to determine trend momentum."""
    try:
        end = datetime.now(timezone.utc) - timedelta(minutes=16) # Avoid real-time subscription requirement
        start = end - timedelta(days=days_back)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        res = data_client.get_stock_bars(req)
        
        history = {}
        for symbol, bars in res.data.items():
            history[symbol] = [{"c": float(b.close), "t": b.timestamp.isoformat()} for b in bars]
        return history
    except Exception as e:
        logging.error(f"[MarketData] Error fetching history data: {e}", exc_info=True)
        return {}
