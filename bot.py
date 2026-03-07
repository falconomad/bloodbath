import os
import json
from google import genai
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockSnapshotRequest
from alpaca.data.screener.screener import ScreenerClient
from alpaca.data.screener.requests import ScreenerRequest
from alpaca.data.screener.filter import ScreenerFilter, ScreenerFilterCondition
import alpaca.data.screener.filter as filters
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
        
        req = ScreenerRequest(
            sort_by=filters.MoversSort.PERCENT_CHANGE,
            direction=filters.SortDirection.DESC,
            limit=limit,
            type=filters.MoversType.GAINERS
        )
        
        response = screener_client.get_top_movers(req)
        
        movers = []
        for mover in response.movers:
            movers.append({
                "symbol": mover.symbol,
                "price": float(mover.price),
                "change_pct": float(mover.percent_change),
                "volume": int(mover.volume)
            })
            
        return movers
        
    except Exception as e:
        print(f"Error fetching top movers via Screener API: {e}")
        return []

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
            
    return {
        "account": {
            "buying_power": buying_power,
            "portfolio_value": portfolio_value,
        },
        "open_positions": current_positions,
        "todays_top_movers": top_movers
    }

def get_ai_recommendation(market_state):
    prompt = f"""
    You are an extremely aggressive day-trading bot. Your goal is to capture 2-5% profits within a 1-2 hour window.
    You do NOT hold stocks long term. 
    
    Current State:
    {json.dumps(market_state, indent=2)}
    
    Rules:
    1. EXITS: If any 'open_positions' have an 'unrealized_pl_pct' greater than 2.0% OR less than -2.0% (cutting losses), you MUST recommend a "sell" action for that entire quantity to free up capital.
    2. ENTRIES: Look at 'todays_top_movers'. Allocate 'buying_power' to buy the highest momentum stocks. Do NOT recommend fractional shares; round down your 'qty' to the nearest whole integer.
    3. Do NOT recommend a trade if there are no good setups or if you just sold a position and need to wait for settlement.
    
    Analyze the current state and provide your recommended trades. You must output valid JSON ONLY, with the following structure:
    {{
        "trades": [
            {{
                "symbol": "TICKER",
                "action": "buy" | "sell",
                "qty": 5,
                "reason": "Brief explanation"
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

def execute_trades(recommendation):
    trades = recommendation.get("trades", [])
    for trade in trades:
        symbol = trade.get("symbol")
        action = trade.get("action")
        qty = trade.get("qty")
        
        if not all([symbol, action, qty]):
            print(f"Invalid trade recommendation: {trade}")
            continue
            
        try:
            # Enforce whole shares for simplicity to avoid fractional DAY order limits
            qty = int(float(qty))
            if qty <= 0:
                print(f"Skipping trade for {symbol} due to 0 quantity after rounding.")
                continue

            side = OrderSide.BUY if action.lower() == 'buy' else OrderSide.SELL
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
    print(f"Recommendation: {json.dumps(recommendation, indent=2)}")
    
    if recommendation and "trades" in recommendation and len(recommendation["trades"]) > 0:
        print("Executing trades...")
        execute_trades(recommendation)
    else:
        print("No trades recommended.")
        
    print("Done.")

if __name__ == "__main__":
    main()
