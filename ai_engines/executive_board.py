import json
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

def make_final_decision(symbol, current_positions, buying_power, technical_report, sentiment_report):
    """
    Synthesizes reports from the Technical and Sentiment Analysts to output a final trade recommendation.
    """
    prompt = f"""
    You are the Chief Investment Officer (CIO) of an aggressive algorithmic day-trading fund.
    Your goal is to increase portfolio value today, never holding overnight.

    Asset: {symbol}
    Current Buying Power: ${buying_power}
    Open Positions: {json.dumps(current_positions)}

    --- ANALYST REPORTS ---
    Technical Analyst: {json.dumps(technical_report)}
    Sentiment Analyst: {json.dumps(sentiment_report)}
    -----------------------

    Rules for ABSOLUTE COMPLIANCE:
    1. STRICT EXITS (Profits): If {symbol} is currently held and 'unrealized_pl_pct' > {config.PROFIT_TAKE_PCT}%, you MUST recommend a "sell" action (allocation_pct 100) to lock in the increment.
    2. STRICT EXITS (Losses): If {symbol} is currently held and 'unrealized_pl_pct' drops below {config.STOP_LOSS_PCT}%, you MUST recommend a "sell" action to cut the loss.
    3. NEW ENTRIES: If we do not hold this stock and you have buying power, evaluate the Analyst Reports. You only want to buy stocks that have BOTH strong upward technical momentum (> 70) AND a bullish fundamental catalyst (> 70).
    4. POSITION SIZING: If buying, output the percentage of your 'Current Buying Power' to allocate to {symbol} (e.g. 50 meaning 50%). Note that the Risk Manager will cap you at {config.MAX_ALLOCATION_PCT}%, so output the raw conviction allocation you desire.

    Output JSON ONLY:
    {{
        "chain_of_thought": "Briefly synthesize the two analyst reports and justify the final execution decision based on the firm's strict rules.",
        "action": "buy" | "sell" | "hold",
        "allocation_pct": <integer 0-100>
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text)
        data['symbol'] = symbol
        return data
    except Exception as e:
        print(f"[ExecutiveBoard] Error synthesizing decision for {symbol}: {e}")
        return {"symbol": symbol, "action": "hold", "allocation_pct": 0, "chain_of_thought": "Executive engine failed."}
