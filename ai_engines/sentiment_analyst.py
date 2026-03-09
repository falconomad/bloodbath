import json
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

def evaluate_sentiment(symbol, news, upcoming_events, macro_calendar):
    """
    Grades a stock strictly on fundamental catalysts and news sentiment.
    Returns a score from 0-100 and a 1-sentence rationale.
    """
    prompt = f"""
    You are a seasoned Macro and Fundamental Sentiment Analyst for a hedge fund.
    
    Macro Calendar: {macro_calendar}
    
    Asset: {symbol}
    Upcoming Corporate Events: {upcoming_events}
    Recent News Headlines: {json.dumps(news)}
    
    Ignore price action. Look only at the catalysts. Is the stock moving on a rumor, 
    a massive government contract, or strong earnings? Is there a geopolitical risk (tariffs, war) 
    threatening this sector right now?
    
    Output JSON ONLY:
    {{
        "score": <0-100 integer representing bullish sentiment conviction>,
        "rationale": "One brief sentence explaining the sentiment score."
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
        print(f"[SentimentAnalyst] Error grading {symbol}: {e}")
        return {"score": 50, "rationale": "Failed to analyze sentiment (Defaulting neutral)."}
