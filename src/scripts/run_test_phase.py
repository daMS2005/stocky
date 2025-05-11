from datetime import datetime, date, timedelta
from src.rag_db import get_vector_store
from src.prediction_logger import PredictionLogger
from src.ingestion.data_fetcher import DataFetcher
import openai
import os
import pandas as pd
from typing import Dict, List
import json
from src.scripts.run_learning_phase import build_prediction_prompt

def evaluate_test_performance(logger: PredictionLogger, ticker: str, start_date: date, end_date: date) -> Dict:
    """Evaluate model performance on test data."""
    predictions = logger.get_past_predictions(ticker, n=None)
    test_predictions = [p for p in predictions if start_date <= p['date'] <= end_date]
    
    if not test_predictions:
        return {
            'accuracy': 0.0,
            'total_predictions': 0,
            'average_return': 0.0,
            'prediction_breakdown': {},
            'return_by_prediction': {},
            'pnl_metrics': {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'sharpe_ratio': 0.0
            }
        }
    
    # Calculate accuracy and returns
    correct_predictions = 0
    prediction_breakdown = {}
    return_by_prediction = {}
    weekly_returns = []
    pnl_metrics = {
        'total_pnl': 0.0,
        'win_rate': 0.0,
        'average_win': 0.0,
        'average_loss': 0.0,
        'largest_win': 0.0,
        'largest_loss': 0.0,
        'sharpe_ratio': 0.0
    }
    
    # Sort predictions by date
    test_predictions.sort(key=lambda x: x['date'])
    
    for i, pred in enumerate(test_predictions):
        if 'actual_return' not in pred:
            continue
            
        actual_return = pred['actual_return']
        prediction = pred['prediction']
        
        # Update prediction breakdown
        prediction_breakdown[prediction] = prediction_breakdown.get(prediction, 0) + 1
        
        # Update return by prediction type
        if prediction not in return_by_prediction:
            return_by_prediction[prediction] = []
        return_by_prediction[prediction].append(actual_return)
        weekly_returns.append(actual_return)
        
        # Update PnL metrics
        pnl_metrics['total_pnl'] += actual_return
        
        if actual_return > 0:
            pnl_metrics['largest_win'] = max(pnl_metrics['largest_win'], actual_return)
        else:
            pnl_metrics['largest_loss'] = min(pnl_metrics['largest_loss'], actual_return)
        
        # Calculate accuracy based on prediction direction
        if i > 0:  # Skip first prediction as we need previous value
            prev_return = test_predictions[i-1].get('actual_return', 0)
            
            # Determine if prediction was correct based on direction
            if prediction in ['STRONG_BUY', 'BUY'] and actual_return > prev_return:
                correct_predictions += 1
            elif prediction in ['STRONG_SELL', 'SELL'] and actual_return < prev_return:
                correct_predictions += 1
            elif prediction == 'HOLD' and abs(actual_return - prev_return) < 0.5:  # Within 0.5% of previous value
                correct_predictions += 1
    
    # Calculate PnL metrics
    if weekly_returns:
        winning_trades = [r for r in weekly_returns if r > 0]
        losing_trades = [r for r in weekly_returns if r < 0]
        
        pnl_metrics['win_rate'] = len(winning_trades) / len(weekly_returns)
        pnl_metrics['average_win'] = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        pnl_metrics['average_loss'] = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0%)
        if len(weekly_returns) > 1:
            returns_std = pd.Series(weekly_returns).std()
            if returns_std > 0:
                pnl_metrics['sharpe_ratio'] = (pnl_metrics['total_pnl'] / len(weekly_returns)) / returns_std
    
    # Calculate average returns by prediction type
    for pred_type in return_by_prediction:
        returns = return_by_prediction[pred_type]
        if returns:
            return_by_prediction[pred_type] = sum(returns) / len(returns)
        else:
            return_by_prediction[pred_type] = 0.0
    
    # Calculate accuracy (excluding first prediction)
    total_predictions = len(test_predictions) - 1
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'total_predictions': total_predictions,
        'average_return': sum(p.get('actual_return', 0) for p in test_predictions) / len(test_predictions),
        'prediction_breakdown': prediction_breakdown,
        'return_by_prediction': return_by_prediction,
        'pnl_metrics': pnl_metrics
    }

def main():
    # Initialize components
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    logger = PredictionLogger()
    data_fetcher = DataFetcher()
    
    # Configuration
    TICKER = "AAPL"
    TEST_START_DATE = datetime(2025, 2, 28).date()  # Start of test phase
    TEST_END_DATE = datetime(2025, 12, 31).date()   # End of test phase
    RESULTS_FILE = "logs/test_results.json"
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Starting Test Phase for {TICKER}")
    print(f"Period: {TEST_START_DATE} to {TEST_END_DATE}")
    print(f"{'='*50}\n")
    
    current_date = TEST_START_DATE
    while current_date <= TEST_END_DATE:
        print(f"\nProcessing week of {current_date}")
        
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
            
            # Get fundamental and macro data
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
            
            # Update previous week's outcome
            if current_date > TEST_START_DATE:
                prev_week = current_date - timedelta(days=7)
                prev_week_start = prev_week - timedelta(days=prev_week.weekday())
                prev_week_end = prev_week_start + timedelta(days=4)
                
                # Get actual return
                prev_df = data_fetcher.fetch_stock_data(TICKER,
                                                      prev_week_start.strftime('%Y-%m-%d'),
                                                      prev_week_end.strftime('%Y-%m-%d'))
                
                if not prev_df.empty:
                    actual_return = float(((prev_df['Close'].iloc[-1] - prev_df['Open'].iloc[0]) / prev_df['Open'].iloc[0]) * 100)
                    
                    # Update outcome without feedback
                    logger.update_outcome(
                        ticker=TICKER,
                        date=prev_week,
                        actual_return=actual_return,
                        feedback="",  # No feedback in test phase
                        reflection=""  # No reflection in test phase
                    )
                    
                    print(f"Updated outcome: {actual_return:.2f}% return")
            
        except Exception as e:
            print(f"Error processing week: {str(e)}")
        
        current_date += timedelta(days=7)
    
    # Evaluate final performance
    print("\nEvaluating test performance...")
    test_results = evaluate_test_performance(logger, TICKER, TEST_START_DATE, TEST_END_DATE)
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest Phase Results:")
    print(f"Accuracy: {test_results['accuracy']:.2%}")
    print(f"Total Predictions: {test_results['total_predictions']}")
    print(f"Average Return: {test_results['average_return']:.2f}%")
    
    print("\nPnL Metrics:")
    pnl = test_results['pnl_metrics']
    print(f"Total PnL: {pnl['total_pnl']:.2f}%")
    print(f"Win Rate: {pnl['win_rate']:.2%}")
    print(f"Average Win: {pnl['average_win']:.2f}%")
    print(f"Average Loss: {pnl['average_loss']:.2f}%")
    print(f"Largest Win: {pnl['largest_win']:.2f}%")
    print(f"Largest Loss: {pnl['largest_loss']:.2f}%")
    print(f"Sharpe Ratio: {pnl['sharpe_ratio']:.2f}")
    
    print("\nPrediction Breakdown:")
    for pred, count in test_results['prediction_breakdown'].items():
        print(f"{pred}: {count}")
    
    print("\nAverage Returns by Prediction:")
    for pred, avg_return in test_results['return_by_prediction'].items():
        print(f"{pred}: {avg_return:.2f}%")
    
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main() 