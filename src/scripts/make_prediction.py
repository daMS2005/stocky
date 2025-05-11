from datetime import datetime, date, timedelta
from src.rag_db import get_vector_store
from src.prediction_logger import PredictionLogger
from src.ingestion.data_fetcher import DataFetcher
import openai
import argparse
import os

def build_prediction_prompt(ticker: str, current_features: dict, similar_past_weeks: list) -> str:
    """Build a prompt for the LLM to make a prediction."""
    
    # Format similar past weeks
    past_weeks_text = "\n".join([
        f"[{week['week_start']}] {week['outcome']} → {week['price_data']['weekly_return']:.2f}% {week.get('feedback', '')}"
        for week in similar_past_weeks
    ])
    
    # Format current features
    features_text = f"""
Current Technical Indicators:
- RSI: {current_features['technical'].get('rsi', 'N/A')}
- MACD: {current_features['technical'].get('macd', 'N/A')}
- Bollinger Bands: {current_features['technical'].get('bollinger_upper', 'N/A')} / {current_features['technical'].get('bollinger_middle', 'N/A')} / {current_features['technical'].get('bollinger_lower', 'N/A')}
- Volume Ratio: {current_features['technical'].get('volume_ratio', 'N/A')}

Current Price Data:
- Open: ${current_features['price_data']['open_price']:.2f}
- Close: ${current_features['price_data']['close_price']:.2f}
- Weekly Return: {current_features['price_data']['weekly_return']:.2f}%

Current Fundamental Data:
- Debt/Equity: {current_features.get('fundamental', {}).get('debt_to_equity', 'N/A')}
- ROE: {current_features.get('fundamental', {}).get('return_on_equity', 'N/A')}
- Beta: {current_features.get('fundamental', {}).get('beta', 'N/A')}

Current Macro Data:
- Country Risk: {current_features.get('macro', {}).get('country_risk', 'N/A')}
- VIX: {current_features.get('macro', {}).get('vix', 'N/A')}
"""
    
    prompt = f"""You are a performance-aware stock analyst. Your goal is to make accurate predictions and learn from past performance.

📁 Related Past Decisions:
{past_weeks_text}

📊 Current Market State:
{features_text}

Based on the current market state and similar past weeks, make a prediction for {ticker} for the next week.
Consider:
1. Technical indicators and their historical reliability
2. Fundamental factors and their current state
3. Macro conditions and their impact
4. Similar past weeks and their outcomes

Provide:
1. Your prediction (STRONG_BUY, BUY, HOLD, SELL, or STRONG_SELL)
2. Your confidence level (0-1)
3. Your reasoning
4. Which features most influenced your decision

Format your response as:
PREDICTION: [your prediction]
CONFIDENCE: [0-1]
REASONING: [your detailed reasoning]
FEATURES: [list of key features used]
"""
    
    return prompt

def main():
    parser = argparse.ArgumentParser(description='Make a weekly prediction for a stock.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--date', type=str, help='Date to make prediction for (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Initialize components
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    logger = PredictionLogger()
    data_fetcher = DataFetcher()
    
    # Get prediction date
    if args.date:
        pred_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        pred_date = date.today()
    
    # Get current week's data
    week_start = pred_date - timedelta(days=pred_date.weekday())
    week_end = week_start + timedelta(days=4)
    
    # Fetch current data
    df = data_fetcher.fetch_stock_data(args.ticker, 
                                     (week_start - timedelta(days=60)).strftime('%Y-%m-%d'),
                                     week_end.strftime('%Y-%m-%d'))
    
    if df.empty:
        print(f"No data available for {args.ticker} on {pred_date}")
        return
    
    # Calculate features
    df = data_fetcher.calculate_technical_indicators(df)
    week_df = df[df.index.date >= week_start]
    
    # Get current features
    current_features = {
        'technical': {
            'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else None,
            'macd': float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else None,
            'bollinger_upper': float(df['BB_Upper'].iloc[-1]) if 'BB_Upper' in df.columns and not pd.isna(df['BB_Upper'].iloc[-1]) else None,
            'bollinger_lower': float(df['BB_Lower'].iloc[-1]) if 'BB_Lower' in df.columns and not pd.isna(df['BB_Lower'].iloc[-1]) else None,
            'bollinger_middle': float(df['BB_Middle'].iloc[-1]) if 'BB_Middle' in df.columns and not pd.isna(df['BB_Middle'].iloc[-1]) else None,
            'volume_ratio': float(df['Volume_Ratio'].iloc[-1]) if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]) else None
        },
        'price_data': {
            'open_price': float(week_df['Open'].iloc[0]),
            'close_price': float(week_df['Close'].iloc[-1]),
            'weekly_return': float(((week_df['Close'].iloc[-1] - week_df['Open'].iloc[0]) / week_df['Open'].iloc[0]) * 100)
        }
    }
    
    # Try to get fundamental and macro data
    try:
        ratios = data_fetcher.get_financial_ratios(args.ticker)
        current_features['fundamental'] = {
            'debt_to_equity': ratios.get('debt_to_equity'),
            'return_on_equity': ratios.get('return_on_equity'),
            'beta': ratios.get('beta')
        }
    except Exception as e:
        print(f"Warning: Could not fetch fundamental data: {str(e)}")
    
    try:
        current_features['macro'] = {
            'country_risk': data_fetcher.get_country_risk(args.ticker),
            'vix': data_fetcher.get_vix_from_fred()
        }
    except Exception as e:
        print(f"Warning: Could not fetch macro data: {str(e)}")
    
    # Get similar past weeks
    similar_weeks = vector_store.query_similar(current_features, k=5)
    
    # Build and send prompt to GPT
    prompt = build_prediction_prompt(args.ticker, current_features, similar_weeks)
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a performance-aware stock analyst. Learn from past predictions and adapt your strategy."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    # Parse response
    response_text = response.choices[0].message.content
    
    # Extract prediction components
    prediction = None
    confidence = None
    reasoning = None
    features = []
    
    for line in response_text.split('\n'):
        if line.startswith('PREDICTION:'):
            prediction = line.split(':', 1)[1].strip()
        elif line.startswith('CONFIDENCE:'):
            confidence = float(line.split(':', 1)[1].strip())
        elif line.startswith('REASONING:'):
            reasoning = line.split(':', 1)[1].strip()
        elif line.startswith('FEATURES:'):
            features = [f.strip() for f in line.split(':', 1)[1].strip().split(',')]
    
    if not all([prediction, confidence, reasoning, features]):
        print("Error: Could not parse GPT response")
        return
    
    # Log prediction
    logger.log_prediction(
        ticker=args.ticker,
        date=pred_date,
        prediction=prediction,
        confidence=confidence,
        reasoning=reasoning,
        features_used=features
    )
    
    print(f"\nPrediction logged for {args.ticker} on {pred_date}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence}")
    print(f"Reasoning: {reasoning}")
    print(f"Key Features: {', '.join(features)}")

if __name__ == "__main__":
    main() 