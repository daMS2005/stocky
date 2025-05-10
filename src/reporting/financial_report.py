from typing import Dict, Any, List
from datetime import datetime

def generate_structured_json(
    ticker: str,
    timeframe: Dict[str, str],
    prediction: Dict[str, Any],
    reasoning: str,
    used_features: List[str],
    prompt_feedback: str,
    date: str,
    current_price: float
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "date": date,
        "timeframe": timeframe,
        "current_price": current_price,
        "prediction": prediction,
        "reasoning": reasoning,
        "used_features": used_features,
        "prompt_feedback": prompt_feedback
    }

def generate_natural_language_report(
    ticker: str,
    timeframe: Dict[str, str],
    prediction: Dict[str, Any],
    reasoning: str,
    used_features: List[str],
    prompt_feedback: str,
    generated_on: str = None,
    current_price: float = None
) -> str:
    generated_on = generated_on or datetime.utcnow().strftime('%B %d, %Y')
    current_price_str = f"${float(current_price):.2f}" if current_price not in (None, "N/A", "") else "N/A"
    report = f"""**Investment Report: {ticker}**  
**Week:** {timeframe['start']}–{timeframe['end']}  
**Generated on:** {generated_on}  
**Current Price:** {current_price_str}\n\n"""
    
    def format_price(value):
        try:
            if value in (None, "N/A", ""):
                return "N/A"
            return f"${float(value):.2f}"
        except Exception:
            return "N/A"
    
    # Format numeric fields with proper handling of None values
    entry_price = format_price(prediction.get('entry_price'))
    target_price = format_price(prediction.get('target_price'))
    stop_loss = format_price(prediction.get('stop_loss'))
    confidence = f"{int(prediction['confidence']*100)}%" if prediction.get('confidence') is not None else "N/A"
    expected_return = f"{prediction['expected_return_pct']:+.2f}%" if prediction.get('expected_return_pct') is not None else "N/A"
    
    report += f"""📊 **Recommendation**  
- Action: {'✅' if prediction.get('action', '').lower() == 'buy' else '❌'} {prediction.get('action', 'N/A')}  
- Entry: {entry_price}  
- Target: {target_price}  
- Stop-loss: {stop_loss}  
- Confidence: {confidence}  
- Expected Return: {expected_return}\n\n"""
    
    report += f"""🧠 **Reasoning**  
{reasoning}\n\n"""
    
    report += f"""📌 **Strategy Notes**\n- Features used: {', '.join(used_features)}\n- Feedback: {prompt_feedback}\n\n"""
    
    report += f"""📈 **Summary Table**\n\n| Metric        | Value     |\n|---------------|-----------|\n| Action        | {prediction.get('action', 'N/A')}       |\n| Entry Price   | {entry_price}   |\n| Target Price  | {target_price}   |\n| Stop-Loss     | {stop_loss}   |\n| Confidence    | {confidence}       |\n"""
    
    return report 