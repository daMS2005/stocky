from typing import Dict
from datetime import date
from src.ingestion.data_fetcher import DataFetcher

# Use the real DataFetcher to get all features

def extract_numerical_features(ticker: str, week_start: date, week_end: date) -> Dict[str, float]:
    """
    Compute and return all actual features for the given ticker and week using DataFetcher.
    Only use data up to week_end (no leakage).
    Returns a dict of normalized features (technical, fundamental, macro, lagged, etc.)
    """
    fetcher = DataFetcher()
    # Convert dates to string for DataFetcher
    start_str = week_start.strftime('%Y-%m-%d')
    end_str = week_end.strftime('%Y-%m-%d')
    features = fetcher.get_all_features(ticker)
    # Flatten and select relevant features
    out = {}
    # Technical
    for k in ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'Volume_MA', 'Volume_Ratio']:
        if k in features.get('technical_signals', {}):
            out[k.lower()] = features['technical_signals'][k]
    # Fundamental
    for k in ['debt_to_equity', 'return_on_equity', 'beta', 'dividend_yield', 'institutional_ownership', 'short_interest_ratio']:
        if k in features.get('fundamental', {}).get('financial_ratios', {}):
            out[k] = features['fundamental']['financial_ratios'][k]
    # Macro/Market
    market = features.get('market_features', {})
    for k in ['country_risk', 'vix', 'garch_volatility']:
        if k in market:
            out[k] = market[k]
    # FX rates (flatten)
    fx = market.get('fx_rates', {})
    for fxk, fxv in fx.items():
        out[f'fx_{fxk.lower()}'] = fxv
    # Commodity prices (flatten)
    commodities = market.get('commodity_prices', {})
    for comk, comv in commodities.items():
        out[f'commodity_{comk.lower()}'] = comv
    # Lagged features (last row in price_data)
    if features.get('price_data'):
        last = features['price_data'][-1]
        for k in ['Close_t-1', 'Return_t-1', 'Close_t-2', 'Return_t-2', 'Close_t-3', 'Return_t-3']:
            if k in last:
                out[k.lower()] = last[k]
    # Sentiment
    if 'combined_sentiment' in features.get('sentiment', {}):
        out['combined_sentiment'] = features['sentiment']['combined_sentiment']
    return out 