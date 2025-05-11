from datetime import datetime, date, timedelta
from src.rag_db import get_vector_store
from src.prediction_logger import PredictionLogger
from src.ingestion.data_fetcher import DataFetcher
import openai
import os
import pandas as pd
from typing import Dict, List
import json

def get_learning_metrics(logger: PredictionLogger, ticker: str) -> Dict:
    """Get current learning metrics."""
    summary = logger.get_performance_summary(ticker)
    
    # Get recent predictions for trend analysis
    recent_preds = logger.get_past_predictions(ticker, n=10)
    
    # Calculate rolling accuracy
    if len(recent_preds) >= 5:
        recent_correct = sum(1 for p in recent_preds[:5] if p['outcome'] == "✅")
        rolling_accuracy = recent_correct / 5
    else:
        rolling_accuracy = None
    
    return {
        'overall_accuracy': summary['accuracy'],
        'rolling_accuracy': rolling_accuracy,
        'total_predictions': summary['total_predictions'],
        'average_return': summary['average_return']
    }

def build_prediction_prompt(ticker: str, current_features: dict, similar_weeks: list) -> str:
    """Build a prompt for the LLM to make a prediction."""
    # Log and skip any non-dict entries
    filtered_weeks = []
    for week in similar_weeks:
        if not isinstance(week, dict):
            print(f"Warning: similar_weeks contains a non-dict: {week}")
            continue
        filtered_weeks.append(week)
    past_weeks_text = "\n".join([
        f"[{week.get('week_start', 'N/A')}] {week.get('outcome', 'N/A')} → {float(week.get('price_data', {}).get('weekly_return', 0.0)):.2f}% {week.get('feedback', '')}"
        for week in filtered_weeks
    ])
    # Handle price_data as dict or list
    price_data = current_features.get('price_data', {})
    if isinstance(price_data, dict):
        open_price = float(price_data.get('open_price', 0.0))
        close_price = float(price_data.get('close_price', 0.0))
        weekly_return = float(price_data.get('weekly_return', 0.0))
    elif isinstance(price_data, list) and price_data and isinstance(price_data[0], dict):
        open_price = float(price_data[0].get('open_price', 0.0))
        close_price = float(price_data[0].get('close_price', 0.0))
        weekly_return = float(price_data[0].get('weekly_return', 0.0))
    else:
        open_price = close_price = weekly_return = 0.0
    # Format current features with guards
    features_text = f"""
Current Technical Indicators:
- RSI: {current_features.get('technical', {}).get('rsi', 'N/A')}
- MACD: {current_features.get('technical', {}).get('macd', 'N/A')}
- Bollinger Bands: {current_features.get('technical', {}).get('bollinger_upper', 'N/A')} / {current_features.get('technical', {}).get('bollinger_middle', 'N/A')} / {current_features.get('technical', {}).get('bollinger_lower', 'N/A')}
- Volume Ratio: {current_features.get('technical', {}).get('volume_ratio', 'N/A')}

Current Price Data:
- Open: ${open_price:.2f}
- Close: ${close_price:.2f}
- Weekly Return: {weekly_return:.2f}%

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

def get_previous_prediction(logger: PredictionLogger, ticker: str, current_date: date) -> str:
    """Get the previous week's prediction."""
    predictions = logger.get_past_predictions(ticker, n=2)  # Get last 2 predictions
    if len(predictions) > 1:
        return predictions[1]['prediction']  # Return the previous week's prediction
    return None

def adjust_prediction(prediction: str, previous_prediction: str) -> str:
    """Adjust prediction based on trend continuation rules."""
    if prediction == "HOLD" and previous_prediction in ["BUY", "SELL"]:
        return previous_prediction  # Continue the trend
    return prediction

def generate_feedback(prediction: str, actual_return: float, previous_prediction: str) -> str:
    """Generate feedback based on prediction accuracy and trend continuation."""
    if prediction == "HOLD" and previous_prediction in ["BUY", "SELL"]:
        # If we continued a trend, evaluate based on the trend
        if previous_prediction == "BUY" and actual_return > 0:
            return "Good trend continuation on the upward movement"
        elif previous_prediction == "SELL" and actual_return < 0:
            return "Good trend continuation on the downward movement"
        else:
            return "Missed the trend reversal"
    else:
        # Regular feedback for non-HOLD predictions
        if prediction in ["STRONG_BUY", "BUY"] and actual_return > 0:
            return "Good call on the upward trend"
        elif prediction in ["STRONG_SELL", "SELL"] and actual_return < 0:
            return "Good call on the downward trend"
        elif prediction == "HOLD" and abs(actual_return) < 0.5:
            return "Good call on the sideways movement"
        else:
            return "Missed the market direction"

def generate_reflection(prediction: str, actual_return: float, previous_prediction: str, features_used: List[str]) -> str:
    """Generate reflection based on prediction outcome and features used."""
    if prediction == "HOLD" and previous_prediction in ["BUY", "SELL"]:
        if (previous_prediction == "BUY" and actual_return > 0) or (previous_prediction == "SELL" and actual_return < 0):
            return "Successfully maintained trend continuation. Consider using trend indicators more prominently."
        else:
            return "Failed to identify trend reversal. Need to improve trend reversal detection."
    else:
        if 'technical' in features_used:
            return "Need to improve on technical analysis"
        elif 'fundamental' in features_used:
            return "Need to improve on fundamental analysis"
        else:
            return "Need to improve on overall analysis"

def main():
    # Initialize components
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    logger = PredictionLogger()
    data_fetcher = DataFetcher()
    
    # Configuration
    TICKER = "AAPL"
    START_DATE = datetime(2022, 6, 30).date()  # Last day of June 2022
    END_DATE = datetime(2025, 2, 28).date()    # End of February 2025
    METRICS_FILE = "logs/learning_metrics.json"
    
    # Create metrics directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize metrics history
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics_history = json.load(f)
    else:
        metrics_history = []
    
    current_date = START_DATE
    while current_date <= END_DATE:
        print(f"\n{'='*50}")
        print(f"Processing week of {current_date}")
        print(f"{'='*50}")
        
        # 1. Make prediction for the week
        print("\n1. Making prediction...")
        try:
            # Get current week's data
            week_start = current_date - timedelta(days=current_date.weekday())
            week_end = week_start + timedelta(days=4)
            
            # Fetch current data
            df = data_fetcher.fetch_stock_data(TICKER, 
                                             (week_start - timedelta(days=60)).strftime('%Y-%m-%d'),
                                             week_end.strftime('%Y-%m-%d'))
            
            if df.empty:
                print(f"No data available for {TICKER} on {current_date}")
                current_date += timedelta(days=7)
                continue
            
            # Calculate features
            df = data_fetcher.calculate_technical_indicators(df)
            week_df = df[df.index.date >= week_start]
            
            # Get current features
            current_features = {
                'technical': {
                    'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 0.0,
                    'macd': float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else 0.0,
                    'bollinger_upper': float(df['BB_Upper'].iloc[-1]) if 'BB_Upper' in df.columns and not pd.isna(df['BB_Upper'].iloc[-1]) else 0.0,
                    'bollinger_lower': float(df['BB_Lower'].iloc[-1]) if 'BB_Lower' in df.columns and not pd.isna(df['BB_Lower'].iloc[-1]) else 0.0,
                    'bollinger_middle': float(df['BB_Middle'].iloc[-1]) if 'BB_Middle' in df.columns and not pd.isna(df['BB_Middle'].iloc[-1]) else 0.0,
                    'volume_ratio': float(df['Volume_Ratio'].iloc[-1]) if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]) else 0.0
                },
                'price_data': {
                    'open_price': float(week_df['Open'].iloc[0]),
                    'close_price': float(week_df['Close'].iloc[-1]),
                    'weekly_return': float(((week_df['Close'].iloc[-1] - week_df['Open'].iloc[0]) / week_df['Open'].iloc[0]) * 100)
                }
            }
            
            # Try to get fundamental and macro data
            try:
                ratios = data_fetcher.get_financial_ratios(TICKER)
                if isinstance(ratios, dict):
                    current_features['fundamental'] = {
                        'debt_to_equity': float(ratios.get('debt_to_equity', 0.0)),
                        'return_on_equity': float(ratios.get('return_on_equity', 0.0)),
                        'beta': float(ratios.get('beta', 0.0))
                    }
                else:
                    current_features['fundamental'] = {
                        'debt_to_equity': 0.0,
                        'return_on_equity': 0.0,
                        'beta': 0.0
                    }
            except Exception as e:
                print(f"Warning: Could not fetch fundamental data: {str(e)}")
                current_features['fundamental'] = {
                    'debt_to_equity': 0.0,
                    'return_on_equity': 0.0,
                    'beta': 0.0
                }
            
            try:
                country_risk = data_fetcher.get_country_risk(TICKER)
                vix = data_fetcher.get_vix_from_fred()
                
                current_features['macro'] = {
                    'country_risk': float(country_risk) if country_risk is not None else 0.0,
                    'vix': float(vix) if vix is not None else 0.0
                }
            except Exception as e:
                print(f"Warning: Could not fetch macro data: {str(e)}")
                current_features['macro'] = {
                    'country_risk': 0.0,
                    'vix': 0.0
                }
            
            # Get similar past weeks
            # Create a flat list of feature values for embedding
            feature_values = []
            for category in ['technical', 'price_data', 'fundamental', 'macro']:
                if category in current_features:
                    for value in current_features[category].values():
                        if isinstance(value, (int, float)):
                            feature_values.append(float(value))
                        elif isinstance(value, dict):
                            for v in value.values():
                                if isinstance(v, (int, float)):
                                    feature_values.append(float(v))
            
            # Pad or truncate to match embedding dimension
            if len(feature_values) < 1536:
                feature_values.extend([0.0] * (1536 - len(feature_values)))
            else:
                feature_values = feature_values[:1536]
            
            similar_weeks = vector_store.query_similar(feature_values, k=5)
            
            # Get previous week's prediction
            previous_prediction = get_previous_prediction(logger, TICKER, current_date)
            
            # Make prediction
            prompt = build_prediction_prompt(TICKER, current_features, similar_weeks)
            
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a performance-aware stock analyst. Learn from past predictions and adapt your strategy. Provide concise, well-formatted responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            print("\nGPT Response:")
            print(response_text)
            
            # Clean up the response text
            response_text = response_text.replace('\n\n', '\n').strip()
            
            # Extract prediction components
            prediction = None
            confidence = None
            reasoning = None
            features = []
            
            # Split into sections and process each
            sections = response_text.split('\n')
            current_section = []
            
            for line in sections:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('PREDICTION:'):
                    prediction = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith('REASONING:'):
                    current_section = []
                elif line.startswith('FEATURES:'):
                    features = [f.strip() for f in line.split(':', 1)[1].strip().split(',')]
                elif prediction and confidence is not None and not features:
                    current_section.append(line)
            
            # Combine reasoning lines
            reasoning = ' '.join(current_section) if current_section else None
            
            if not all([prediction, confidence is not None, reasoning, features]):
                print("Error: Could not parse GPT response")
                print("Missing components:")
                if not prediction: print("- Prediction")
                if confidence is None: print("- Confidence")
                if not reasoning: print("- Reasoning")
                if not features: print("- Features")
                current_date += timedelta(days=7)
                continue
            
            # Adjust prediction based on trend continuation
            adjusted_prediction = adjust_prediction(prediction, previous_prediction)
            if adjusted_prediction != prediction:
                print(f"\nAdjusted prediction from {prediction} to {adjusted_prediction} based on trend continuation")
                prediction = adjusted_prediction
            
            # Log prediction
            logger.log_prediction(
                ticker=TICKER,
                date=current_date,
                prediction=prediction,
                confidence=confidence,
                reasoning=reasoning,
                features_used=features
            )
            
            print(f"\nPrediction logged:")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence}")
            print(f"Reasoning: {reasoning}")
            print(f"Features used: {', '.join(features)}")
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            current_date += timedelta(days=7)
            continue
        
        # 2. Update previous week's outcome
        if current_date > START_DATE:
            print("\n2. Updating previous week's outcome...")
            try:
                prev_week = current_date - timedelta(days=7)
                prev_week_start = prev_week - timedelta(days=prev_week.weekday())
                prev_week_end = prev_week_start + timedelta(days=4)
                
                # Get actual return
                prev_df = data_fetcher.fetch_stock_data(TICKER,
                                                      prev_week_start.strftime('%Y-%m-%d'),
                                                      prev_week_end.strftime('%Y-%m-%d'))
                
                if not prev_df.empty:
                    actual_return = float(((prev_df['Close'].iloc[-1] - prev_df['Open'].iloc[0]) / prev_df['Open'].iloc[0]) * 100)
                    
                    # Get the prediction for analysis
                    prev_pred = logger.get_past_predictions(TICKER, n=1)[0]
                    prev_prev_pred = get_previous_prediction(logger, TICKER, prev_week)
                    
                    # Generate feedback and reflection based on adjusted prediction
                    feedback = generate_feedback(prev_pred['prediction'], actual_return, prev_prev_pred)
                    reflection = generate_reflection(prev_pred['prediction'], actual_return, prev_prev_pred, prev_pred['features_used'])
                    
                    # Update outcome
                    logger.update_outcome(
                        ticker=TICKER,
                        date=prev_week,
                        actual_return=actual_return,
                        feedback=feedback,
                        reflection=reflection
                    )
                    
                    print(f"Updated outcome: {actual_return:.2f}% return")
                    print(f"Feedback: {feedback}")
                    print(f"Reflection: {reflection}")
            
            except Exception as e:
                print(f"Error updating outcome: {str(e)}")
        
        # 3. Update learning metrics
        print("\n3. Updating learning metrics...")
        try:
            metrics = get_learning_metrics(logger, TICKER)
            metrics['date'] = current_date.strftime('%Y-%m-%d')
            metrics_history.append(metrics)
            
            # Save metrics
            with open(METRICS_FILE, 'w') as f:
                json.dump(metrics_history, f, indent=2)
            
            print(f"Current accuracy: {metrics['overall_accuracy']:.2%}")
            if metrics['rolling_accuracy']:
                print(f"Rolling accuracy: {metrics['rolling_accuracy']:.2%}")
        
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")
        
        # Move to next week
        current_date += timedelta(days=7)
    
    print("\nLearning phase complete!")
    print(f"Final metrics saved to {METRICS_FILE}")

if __name__ == "__main__":
    main() 