import os
import json
import google.generativeai as genai
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# API Keys
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY or not GEMINI_API_KEY:
    print("Missing API keys. Please set ALPACA_API_KEY, ALPACA_SECRET_KEY, and GEMINI_API_KEY.")
    exit(1)

# Initialize Clients
# Note: we use paper=True by default for safety. Set paper=False for live trading if desired.
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Trading parameters
WATCHLIST = ['NVDA', 'TSLA', 'AMD', 'AAPL', 'COIN', 'META']

def get_market_state():
    # Get account info
    account = trading_client.get_account()
    buying_power = float(account.buying_power)
    portfolio_value = float(account.portfolio_value)
    
    # Get positions
    positions = trading_client.get_all_positions()
    current_positions = []
    for p in positions:
        current_positions.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl)
        })
        
    # Get latest quotes
    quote_req = StockLatestQuoteRequest(symbol_or_symbols=WATCHLIST)
    quotes = data_client.get_stock_latest_quote(quote_req)
    
    market_prices = {}
    for symbol in WATCHLIST:
        if symbol in quotes:
            market_prices[symbol] = {
                "ask_price": float(quotes[symbol].ask_price),
                "bid_price": float(quotes[symbol].bid_price)
            }
            
    return {
        "account": {
            "buying_power": buying_power,
            "portfolio_value": portfolio_value,
        },
        "positions": current_positions,
        "market_prices": market_prices
    }

def get_ai_recommendation(market_state):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an aggressive algorithmic trading bot. You need to make quick, profitable decisions based on the current market state and my portfolio.
    Your goal is to maximize short-term profit.
    
    Current State:
    {json.dumps(market_state, indent=2)}
    
    Analyze the current state and provide your recommended trades. You must output valid JSON ONLY, with the following structure:
    {{
        "trades": [
            {{
                "symbol": "TICKER",
                "action": "buy" | "sell",
                "qty": 1.5,
                "reason": "Brief explanation"
            }}
        ]
    }}
    
    Respond ONLY with the JSON code block. Do not include markdown formatting or any other text before or after the JSON.
    """
    
    response = model.generate_content(prompt)
    try:
        # Strip markdown if present
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
            side = OrderSide.BUY if action.lower() == 'buy' else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=symbol,
                qty=float(qty),
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
    
    print("Executing trades...")
    execute_trades(recommendation)
    print("Done.")

if __name__ == "__main__":
    main()
