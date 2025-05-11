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
    current_price: float,
    features: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Generate structured JSON output including market and macro features."""
    structured = {
        "ticker": ticker,
        "date": timeframe['start'],
        "timeframe": timeframe,
        "current_price": current_price,
        "prediction": prediction,
        "reasoning": reasoning,
        "used_features": used_features,
        "prompt_feedback": prompt_feedback
    }
    
    if features:
        # Add all top-level market features for backward compatibility
        structured.update({
            "fx_rates": features.get('fx_rates', {}),
            "country_risk": features.get('country_risk'),
            "vix": features.get('vix'),
            "garch_volatility": features.get('garch_volatility'),
            "commodity_prices": features.get('commodity_prices', {}),
            "company_commodities": features.get('company_commodities', [])
        })
        # Also add the new market_features dict if present
        if 'market_features' in features:
            structured["market_features"] = features["market_features"]
        # Always add macro_indicators if present, with robust debug prints
        macro_indicators = None
        try:
            macro_indicators = features['fundamental']['macro_indicators']
            print('DEBUG: macro_indicators in features:', macro_indicators)
        except Exception as e:
            print('DEBUG: Could not find macro_indicators:', e)
        if macro_indicators is not None:
            structured['macro_indicators'] = macro_indicators
            print('DEBUG: macro_indicators added to structured:', structured['macro_indicators'])
        else:
            print('DEBUG: macro_indicators NOT FOUND in features')
    print('DEBUG: Final structured JSON:', structured)
    return structured

def generate_natural_language_report(
    ticker: str,
    timeframe: Dict[str, str],
    prediction: Dict[str, Any],
    reasoning: str,
    used_features: List[str],
    prompt_feedback: str,
    generated_on: str = None,
    current_price: float = None,
    features: Dict[str, Any] = None
) -> str:
    """Generate natural language report including market and macro features."""
    generated_on = generated_on or datetime.utcnow().strftime('%B %d, %Y')
    current_price_str = f"${float(current_price):.2f}" if current_price not in (None, "N/A", "") else "N/A"
    
    # Extract market features from the features dictionary
    market_features = features.get('market_features', {}) if features else {}
    fx_rates = market_features.get('fx_rates', {})
    country_risk = market_features.get('country_risk', 'N/A')
    commodity_prices = market_features.get('commodity_prices', {})
    company_commodities = market_features.get('company_commodities', [])
    vix = market_features.get('vix', 'N/A')
    garch_volatility = market_features.get('garch_volatility', 'N/A')
    
    # Format lagged features
    lagged_features = []
    if features and 'price_data' in features and features['price_data']:
        price_data = features['price_data']
        for lag in range(1, 4):
            close = price_data[-1].get(f'Close_t-{lag}', 'N/A')
            ret = price_data[-1].get(f'Return_t-{lag}', 'N/A')
            lagged_features.append(f"Close_t-{lag}: {close}, Return_t-{lag}: {ret}")
    lagged_features_summary = '\n'.join(lagged_features)
    
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
- Action: {'✅' if prediction.get('action', '').lower() == 'buy' else '❌' if prediction.get('action', '').lower() == 'sell' else '⚠️'} {prediction.get('action', 'N/A')}  
- Entry: {entry_price}  
- Target: {target_price}  
- Stop-loss: {stop_loss}  
- Confidence: {confidence}  
- Expected Return: {expected_return}\n\n"""
    
    report += f"""🧠 **Reasoning**  
{reasoning}\n\n"""
    
    report += f"""📌 **Strategy Notes**\n- Features used: {', '.join(used_features)}\n- Feedback: {prompt_feedback}\n\n"""
    
    report += f"""📈 **Summary Table**\n\n| Metric        | Value     |\n|---------------|-----------|\n| Action        | {prediction.get('action', 'N/A')}       |\n| Entry Price   | {entry_price}   |\n| Target Price  | {target_price}   |\n| Stop-Loss     | {stop_loss}   |\n| Confidence    | {confidence}       |\n"""
    
    # Add market and macro features section
    report += f"""\n---\n\n**Additional Market & Macro Features**\n"""
    
    # Format FX rates
    fx_str = ", ".join([f"{k}: {v:.4f}" for k, v in fx_rates.items()]) if fx_rates else "N/A"
    report += f"- FX Rates: {fx_str}\n"
    
    # Format country risk
    report += f"- Country Risk Index: {country_risk}\n"
    
    # Format commodities
    commodity_list = ", ".join(company_commodities) if company_commodities else "N/A"
    report += f"- Relevant Commodities: {commodity_list}\n"
    
    # Format commodity prices
    commodity_str = ", ".join([
        f"{k}: ${v:.2f}" if v is not None else f"{k}: N/A" for k, v in commodity_prices.items()
    ]) if commodity_prices else "N/A"
    report += f"- Commodity Prices: {commodity_str}\n"
    
    # Format volatility indices
    report += f"- Volatility Indices: VIX={vix}, GARCH={garch_volatility}\n"
    
    # Format lagged features
    if lagged_features:
        report += f"- Lagged Features (recent days):\n{lagged_features_summary}\n"
    
    return report 