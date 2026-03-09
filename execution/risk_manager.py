import config

def validate_and_size_trade(trade_recommendation, market_state):
    """
    Acts as a hardcoded 'Kill Switch' before sending any orders to Alpaca.
    - Halts trading if the broad market (SPY) is crashing.
    - Forces caps on max allocations to prevent AI hallucinations placing 100% in a random ticker.
    """
    symbol = trade_recommendation.get("symbol")
    action = trade_recommendation.get("action", "").lower()
    raw_alloc_pct = trade_recommendation.get("allocation_pct", 0)
    
    if action == "hold" or raw_alloc_pct <= 0:
        print(f"[RiskManager] Trade for {symbol} rejected: Action is HOLD or allocation is 0.")
        return None
        
    buying_power = market_state.get("account", {}).get("buying_power", 0)
    
    # 1. Macro Crash Protection
    spy_history = market_state.get("broad_market_spy", [])
    if spy_history and len(spy_history) >= 2:
        today_close = spy_history[-1].get("c", 1)
        prev_close = spy_history[-2].get("c", 1)
        spy_drop = ((today_close - prev_close) / prev_close) * 100
        if spy_drop <= config.MACRO_DOWNTURN_LIMIT and action == "buy":
             print(f"[RiskManager] Trade for {symbol} rejected: SPY is crashing ({spy_drop:.2f}%). Halting BUYS.")
             return None

    # 2. Enforce Max Allocation Caps for new Buy positions
    capped_alloc_pct = min(raw_alloc_pct, config.MAX_ALLOCATION_PCT) if action == "buy" else raw_alloc_pct
    
    if action == "buy" and capped_alloc_pct < raw_alloc_pct:
        print(f"[RiskManager] {symbol} allocation capped from {raw_alloc_pct}% to {capped_alloc_pct}%")

    target_value = buying_power * (capped_alloc_pct / 100.0)

    # 3. Prevent Fractional Share Violations (Calculate integer quantity)
    current_price = None
    for mover in market_state.get("todays_top_movers", []):
        if mover.get("symbol") == symbol:
            current_price = mover.get("price")
            break
            
    # Fallback to portfolio positions if not in top movers
    if not current_price:
         for p in market_state.get("open_positions", []):
             if p.get("symbol") == symbol:
                 current_price = p.get("current_price")
                 break

    if action == "buy":
        if not current_price:
            print(f"[RiskManager] Trade for {symbol} rejected: Could not determine current price.")
            return None
        qty = int(target_value / current_price)
        if qty <= 0:
            print(f"[RiskManager] Trade for {symbol} rejected: Targeted value (${target_value:.2f}) is too small to buy 1 share.")
            return None
    else:
        # If selling, the executor will just call `close_position()` so qty is not strictly needed
        qty = 0 

    return {
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "capped_alloc_pct": capped_alloc_pct
    }
