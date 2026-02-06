# bloodbath
AI-powered stock analysis tool that detects major price dips and evaluates buy opportunities using free market data and news sentiment analysis.

# Stock Dip Analyzer

Stock Dip Analyzer is a lightweight Python-based application that identifies potential stock buying opportunities by combining market data analysis and news sentiment.

The goal of this project is to help investors detect major stock price drops (e.g., 10–20%) and evaluate whether the dip represents a good buying opportunity based on trends, recent news, and market sentiment.

## Features

- Fetch real-time and historical stock data from free APIs  
- Detect stocks that have dropped significantly in recent days  
- Analyze recent news headlines related to a stock  
- Perform sentiment analysis on news articles  
- Combine price trends and sentiment to generate basic buy/sell insights  
- Fully automated pipeline that can run on-demand or on page load  
- Designed to work with free, publicly available data sources  
- Simple architecture that can be deployed easily to Netlify or similar platforms  

## How It Works

1. The script retrieves stock price data using free financial APIs.  
2. It compares current price with recent highs to detect significant drops.  
3. If a stock is down beyond a defined threshold, the system gathers related news articles.  
4. Sentiment analysis is performed on the news to understand market perception.  
5. A recommendation is generated based on:
   - Percentage drop  
   - Market trend  
   - News sentiment  
   - Upcoming events such as earnings calls  

## Technology Stack

- Python  
- Free stock market APIs (e.g., Alpha Vantage, Yahoo Finance)  
- News APIs or web scraping  
- NLP-based sentiment analysis  
- Pandas and Requests libraries  

## Use Cases

- Finding “buy the dip” opportunities  
- Tracking market trends without paid tools  
- Automating daily market research  
- Learning project for financial data analysis  

## Disclaimer

This project is for educational and informational purposes only.  
It does not provide financial advice and should not be used as the sole basis for investment decisions.

---

Contributions, suggestions, and improvements are welcome!
