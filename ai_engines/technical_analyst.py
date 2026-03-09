import json
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

def evaluate_technicals(symbol, history, spy_context):
    """
    Grades a stock strictly on technical trends (price action, momentum)
    Returns a score from 0-100 and a 1-sentence rationale.
    """
    prompt = f"""
    You are a purely quantitative Technical Analyst.
    
    Overall Market Trend (SPY last 5 days): {json.dumps(spy_context)}
    
    Asset: {symbol}
    Price History (Last 5 days): {json.dumps(history)}
    
    Ignore any fundamentals. Look only at the price action. Is it a dead-cat bounce or a sustained breakout? 
    Is it outperforming SPY?
    
    Output JSON ONLY:
    {{
        "score": <0-100 integer representing conviction of an uptrend>,
        "rationale": "One brief sentence explaining the technical score."
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        print(f"[TechnicalAnalyst] Error grading {symbol}: {e}")
        return {"score": 0, "rationale": "Failed to analyze technicals."}
