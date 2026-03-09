import json
import time
import config
from data_ingestion import market_data, news_feed, calendar_events
from ai_engines import technical_analyst, sentiment_analyst, executive_board
from execution import risk_manager, trade_executor
from alpaca.trading.client import TradingClient

def main():
    print("="*50)
    print("🚀 INITIALIZING BLOODBATH MULTI-ENGINE PIPELINE 🚀")
    print("="*50)
    
    trading_client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, paper=config.PAPER_TRADING)
    
    # ---------------------------------------------------------
    # 1. FETCH MARKET STATE
    # ---------------------------------------------------------
    print("\n[Stage 1] Fetching Broad Market State...")
    try:
        account = trading_client.get_account()
        buying_power = float(account.buying_power)
        print(f"💰 Available Buying Power: ${buying_power:,.2f}")
        
        positions = trading_client.get_all_positions()
        current_positions = []
        for p in positions:
            current_positions.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl_pct": float(p.unrealized_plpc) * 100,
                "current_price": float(p.current_price)
            })
        print(f"📊 Open Positions: {len(current_positions)}")
        for pos in current_positions:
            print(f"   - {pos['symbol']}: {pos['unrealized_pl_pct']:.2f}% P/L")
            
    except Exception as e:
        print(f"❌ Failed to fetch account state: {e}")
        return

    # ---------------------------------------------------------
    # 2. FETCH TOP MOVERS & HISTORICAL CONTEXT
    # ---------------------------------------------------------
    print("\n[Stage 2] Scanning for Top Market Movers...")
    top_movers = market_data.get_top_movers(limit=15)
    print(f"🔍 Found {len(top_movers)} volatile movers.")
    
    symbols_to_fetch = ["SPY"] + [m["symbol"] for m in top_movers]
    print(f"📈 Fetching 5-day historical technical data for {len(symbols_to_fetch)} symbols...")
    history = market_data.get_history_context(symbols_to_fetch)
    spy_context = history.pop("SPY", [])
    
    print("\n[Filter] Vetting candidates by liquidity and price...")
    candidates = []
    for mover in top_movers:
        sym = mover["symbol"]
        bars = history.get(sym, [])
        if not bars:
            continue
            
        # Average volume over last 5 days
        avg_vol = sum(b['v'] for b in bars) / len(bars)
        dollar_vol = avg_vol * mover["price"]
        
        if mover["price"] >= config.MIN_PRICE and dollar_vol >= config.MIN_DOLLAR_VOLUME:
            candidates.append((dollar_vol, mover))
            
    # Sort by highest dollar volume to prioritize the most liquid market movers
    candidates.sort(reverse=True, key=lambda x: x[0])
    top_movers = [c[1] for c in candidates[:config.MAX_CANDIDATES]]
    print(f"🎯 Filtered down to {len(top_movers)} absolute best candidates based on ${config.MIN_PRICE}+ price and ${config.MIN_DOLLAR_VOLUME:,.0f}+ 5-day volume.")
    
    if len(top_movers) == 0:
        print("🤷 No candidates met the strict liquidity requirements today. Sleeping until next run.")
        return
    
    # ---------------------------------------------------------
    # 3. FETCH NEWS & SENTIMENT CONTEXT
    # ---------------------------------------------------------
    print("\n[Stage 3] Fetching Fundamental Catalysts...")
    mover_symbols = [m["symbol"] for m in top_movers]
    
    print("📰 Pulling Alpaca News headlines...")
    news_context = news_feed.get_news_context(mover_symbols)
    
    print("🗓️ Pulling Corporate Calendars...")
    upcoming_events = calendar_events.get_earnings_calendar(mover_symbols)
    macro_calendar = calendar_events.get_macro_calendar()

    market_state = {
        "account": {"buying_power": buying_power},
        "open_positions": current_positions,
        "broad_market_spy": spy_context,
        "todays_top_movers": top_movers
    }

    # ---------------------------------------------------------
    # 4. RUN AI ENGINES PER SYMBOL
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🧠 ENGAGING AI COUNCIL FOR INDIVIDUAL ASSETS 🧠")
    print("="*50)
    
    final_recommendations = []
    
    for mover in top_movers:
        symbol = mover["symbol"]
        print(f"\n--- Analyzing: {symbol} ---")
        
        sym_history = history.get(symbol, [])
        sym_news = news_context.get(symbol, [])
        sym_events = upcoming_events.get(symbol, "None")
        
        if not sym_history:
             print(f"   ⚠️ Skipping {symbol} due to missing price history.")
             continue
        
        # Engine 1: Technical
        print("   🤖 Running Technical Analyst...")
        tech_report = technical_analyst.evaluate_technicals(symbol, sym_history, spy_context)
        print(f"      -> Score: {tech_report.get('score', 0)}/100 | {tech_report.get('rationale', 'No rationale')}")
        
        # Engine 2: Sentiment
        print("   🤖 Running Sentiment Analyst...")
        sent_report = sentiment_analyst.evaluate_sentiment(symbol, sym_news, sym_events, macro_calendar)
        print(f"      -> Score: {sent_report.get('score', 0)}/100 | {sent_report.get('rationale', 'No rationale')}")
        
        # Engine 3: Executive Board
        print("   👑 Running Executive Board Synthesis...")
        exec_decision = executive_board.make_final_decision(symbol, current_positions, buying_power, tech_report, sent_report)
        print(f"      -> Action: {exec_decision.get('action', 'HOLD').upper()} | Alloc: {exec_decision.get('allocation_pct', 0)}%")
        print(f"      -> Reasoning: {exec_decision.get('chain_of_thought', '')}")
        
        final_recommendations.append(exec_decision)
        
        print(f"   ⏱️ Sleeping {config.API_SLEEP_SECONDS}s to respect Gemini API Rate Limits...")
        time.sleep(config.API_SLEEP_SECONDS)

    # ---------------------------------------------------------
    # 5. RISK MANAGEMENT & EXECUTION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🛡️ EXECUTING TRADES THROUGH RISK MANAGER 🛡️")
    print("="*50)
    
    trades_executed = 0
    for rec in final_recommendations:
        validated_trade = risk_manager.validate_and_size_trade(rec, market_state)
        if validated_trade:
            order = trade_executor.execute_order(validated_trade)
            if order:
                trades_executed += 1
                
    print(f"\n✅ Pipeline Complete. Executed {trades_executed} trades today.")

if __name__ == "__main__":
    main()
