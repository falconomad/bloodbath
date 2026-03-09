import os
import json
from google import genai
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockSnapshotRequest
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest
from alpaca.data.models.screener import MarketType
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta, timezone

# API Keys
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not ALPACA_API_KEY or not ALPACA_API_SECRET or not GEMINI_API_KEY:
    print("Missing API keys. Please set ALPACA_API_KEY, ALPACA_API_SECRET, and GEMINI_API_KEY.")
    exit(1)

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
screener_client = ScreenerClient(ALPACA_API_KEY, ALPACA_API_SECRET)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def get_top_movers(limit=15):
    """
    Dynamically scan the market for the top gainers of the day.
    """
    try:
        # We can construct a simple screener request.
        # Alternatively, if screener is restrictive, we can fetch all snapshots
        # However, Alpaca's Screener client provides `get_top_movers` endpoints.
        # Let's utilize the native Top Movers screener:
        
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
        print(f"Error fetching top movers via Screener API: {e}")
        return []

def get_history_context(symbols):
    try:
        end = datetime.now(timezone.utc) - timedelta(minutes=16) # Avoid real-time subscription requirement
        start = end - timedelta(days=5)
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
        print(f"Error fetching history data: {e}")
        return {}
def get_market_state():
    # Get account info
    account = trading_client.get_account()
    buying_power = float(account.buying_power)
    portfolio_value = float(account.portfolio_value)
    
    # Get positions with time held
    positions = trading_client.get_all_positions()
    current_positions = []
    
    # We don't have exact 'time opened' natively in the position object without querying orders,
    # so we'll pass the unrealized PL. The bot should sell aggressively if PL > 2% or < -2%
    for p in positions:
        current_positions.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_pl_pct": float(p.unrealized_plpc) * 100, # Convert to percentage
            "current_price": float(p.current_price)
        })
        
    # Get today's top dynamic movers
    top_movers = get_top_movers()
    
    # Get 3-day history context for the top movers (and SPY for broad market)
    symbols_to_fetch = ["SPY"] + [m["symbol"] for m in top_movers]
    history = get_history_context(symbols_to_fetch)
    
    spy_context = history.pop("SPY", [])
    
    # Inject history into movers
    for m in top_movers:
        m["history"] = history.get(str(m.get("symbol")), [])
            
    return {
        "account": {
            "buying_power": buying_power,
            "portfolio_value": portfolio_value,
        },
        "open_positions": current_positions,
        "broad_market_spy": spy_context,
        "todays_top_movers": top_movers
    }

def get_ai_recommendation(market_state):
    prompt = f"""
    You are an extremely aggressive, purely algorithmic day-trading bot. Your ONLY goal is to make sure money is increased today. 
    You are strictly a day trader. We should NEVER have a day with losses, and we DO NOT hold positions overnight or long term.
    
    Current State:
    {json.dumps(market_state, indent=2)}
    
    Rules for ABSOLUTE COMPLIANCE:
    1. STRICT EXITS (Profits): If any 'open_positions' have an 'unrealized_pl_pct' greater than 1.5%, you MUST immediately recommend a "sell" action for the ENTIRE quantity to lock in the 1-2% increment. Reinvest the cash in the next cycle.
    2. STRICT EXITS (Losses): If any 'open_positions' drop below -1.0%, you MUST immediately recommend a "sell" action for the ENTIRE quantity to ruthlessly cut the loss. No hoping for a bounce.
    3. AGGRESSIVE ENTRIES: Look at 'todays_top_movers' which contains both top gainers and top losers. It doesn't matter if it's a popular tech stock like Apple/Google or from another sector. If you have 'buying_power', evaluate them for strong growth or rebound potential. Read their 'history' prices. If there's a strong candidate, recommend a "buy" to increase our money. 
    4. POSITION SIZING: Instead of specific quantities, use an allocation percentage of your available 'buying_power' (e.g., 50 means use 50% of buying power). Do NOT allocate more than 50% of your buying power to a single new ticker in this cycle. If recommending a "sell", set allocation_pct to 100 to dump the whole position.
    5. END OF DAY: If the current time is nearing market close, you MUST sell all open positions regardless of profit/loss.
    
    Analyze the current state and provide your recommended trades. You must explain your chain-of-thought reasoning first, then provide the trades. You must output valid JSON ONLY, with the following structure:
    {{
        "chain_of_thought": "Briefly evaluate SPY, the general top movers history, and justify the selected trades mapping back to the absolute compliance rules.",
        "trades": [
            {{
                "symbol": "TICKER",
                "action": "buy" | "sell",
                "allocation_pct": 50
            }}
        ]
    }}
    
    Respond ONLY with the JSON code block. Do not include markdown formatting or any other text before or after the JSON.
    """
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        return json.loads(text)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
             print(f"Raw response: {response.text}")
        return {"trades": []}

def execute_trades(recommendation, market_state):
    trades = recommendation.get("trades", [])
    buying_power = market_state.get("account", {}).get("buying_power", 0)
    
    for trade in trades:
        symbol = trade.get("symbol")
        action = trade.get("action")
        alloc_pct = trade.get("allocation_pct")
        
        if not all([symbol, action, alloc_pct]):
            print(f"Invalid trade recommendation (missing fields): {trade}")
            continue
            
        try:
            if action.lower() == 'sell':
                # Alpaca provides close_position
                order = trading_client.close_position(symbol)
                print(f"Successfully submitted order: CLOSE position {symbol}")
                continue
                
            # For buying, calculate qty from allocation percentage
            if action.lower() == 'buy':
                # find the current price of the symbol
                current_price = None
                for mover in market_state.get("todays_top_movers", []):
                    if mover.get("symbol") == symbol:
                        current_price = mover.get("price")
                        break
                        
                if not current_price:
                    print(f"Could not find current price for buy target {symbol}. Skipping.")
                    continue
                    
                target_value = buying_power * (alloc_pct / 100.0)
                qty = int(target_value / current_price)
                
                if qty <= 0:
                    print(f"Skipping trade for {symbol} due to 0 quantity after rounding (target_value: {target_value}, price: {current_price}).")
                    continue
    
                side = OrderSide.BUY
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            order = trading_client.submit_order(req)
            print(f"Successfully submitted order: {action.upper()} {qty} of {symbol}")
        except Exception as e:
            print(f"Error executing trade {trade}: {e}")

def main():
    print("Fetching market state...")
    try:
        state = get_market_state()
    except Exception as e:
        print(f"Failed to fetch market state: {e}")
        return

    print("Asking Gemini for recommendations...")
    recommendation = get_ai_recommendation(state)
    print(f"Reasoning: {recommendation.get('chain_of_thought', 'No reasoning provided.')}")
    print(f"Recommendation: {json.dumps(recommendation.get('trades', []), indent=2)}")
    
    if recommendation and "trades" in recommendation and len(recommendation["trades"]) > 0:
        print("Executing trades...")
        execute_trades(recommendation, state)
    else:
        print("No trades recommended.")
        
    print("Done.")

if __name__ == "__main__":
    main()
