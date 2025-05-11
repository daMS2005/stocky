from datetime import datetime, date
import json
import os
from typing import Dict, List, Optional, Tuple

class PredictionLogger:
    def __init__(self, log_dir: str = "logs/predictions"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def log_prediction(self, 
                      ticker: str,
                      date: date,
                      prediction: str,
                      confidence: float,
                      reasoning: str,
                      features_used: List[str]) -> None:
        """Log a new prediction."""
        log_entry = {
            'ticker': ticker,
            'date': date.strftime('%Y-%m-%d'),
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': reasoning,
            'features_used': features_used,
            'outcome': None,  # Will be updated when we know the actual return
            'return': None,
            'feedback': None,
            'reflection': None
        }
        
        # Save to file
        filename = f"{self.log_dir}/{ticker}_{date.strftime('%Y-%m-%d')}.json"
        with open(filename, 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def update_outcome(self,
                      ticker: str,
                      date: date,
                      actual_return: float,
                      feedback: str,
                      reflection: Optional[str] = None) -> None:
        """Update a prediction with its outcome and feedback."""
        filename = f"{self.log_dir}/{ticker}_{date.strftime('%Y-%m-%d')}.json"
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No prediction found for {ticker} on {date}")
        
        with open(filename, 'r') as f:
            log_entry = json.load(f)
        
        # Update with outcome
        log_entry['outcome'] = self._calculate_outcome(log_entry['prediction'], actual_return)
        log_entry['return'] = actual_return
        log_entry['feedback'] = feedback
        if reflection:
            log_entry['reflection'] = reflection
        
        # Save updated entry
        with open(filename, 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def get_past_predictions(self,
                           ticker: str,
                           n: int = 5,
                           include_outcomes: bool = True) -> List[Dict]:
        """Get the n most recent predictions for a ticker."""
        predictions = []
        
        # Get all prediction files for the ticker
        files = [f for f in os.listdir(self.log_dir) if f.startswith(ticker)]
        files.sort(reverse=True)  # Most recent first
        
        for file in files[:n]:
            with open(os.path.join(self.log_dir, file), 'r') as f:
                pred = json.load(f)
                if include_outcomes or pred['outcome'] is not None:
                    predictions.append(pred)
        
        return predictions
    
    def _calculate_outcome(self, prediction: str, actual_return: float) -> str:
        """Calculate if the prediction was correct based on the actual return."""
        if prediction == "STRONG_BUY" and actual_return > 2.0:
            return "✅"
        elif prediction == "BUY" and actual_return > 0.5:
            return "✅"
        elif prediction == "HOLD" and -0.5 <= actual_return <= 0.5:
            return "✅"
        elif prediction == "SELL" and actual_return < -0.5:
            return "✅"
        elif prediction == "STRONG_SELL" and actual_return < -2.0:
            return "✅"
        else:
            return "❌"
    
    def get_performance_summary(self, ticker: str) -> Dict:
        """Get a summary of prediction performance for a ticker."""
        predictions = self.get_past_predictions(ticker, n=1000)  # Get all predictions
        
        total = len(predictions)
        correct = sum(1 for p in predictions if p['outcome'] == "✅")
        
        return {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': correct / total if total > 0 else 0,
            'average_return': sum(p['return'] for p in predictions if p['return'] is not None) / total if total > 0 else 0
        } 