from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import config

trading_client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, paper=config.PAPER_TRADING)

def execute_order(validated_trade):
    """
    Submits a finalized and mathematically validated trade order to Alpaca.
    """
    if not validated_trade:
        return
        
    symbol = validated_trade.get("symbol")
    action = validated_trade.get("action")
    qty = validated_trade.get("qty")
    
    try:
        if action == 'sell':
            # Liquidate the entire open position
            order = trading_client.close_position(symbol)
            print(f"[TradeExecutor] Successfully submitted CLOSE order for position {symbol}")
            return order
            
        if action == 'buy':
            side = OrderSide.BUY
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            order = trading_client.submit_order(req)
            print(f"[TradeExecutor] Successfully submitted BUY order: {qty} shares of {symbol}")
            return order
    except Exception as e:
        print(f"[TradeExecutor] Failed to execute {action} order for {symbol}: {e}")
        return None
