import os
from datetime import datetime
import pandas as pd
from src.ingestion.data_fetcher import DataFetcher
import numpy as np

def test_market_features():
    """Test individual market and macro feature fetchers."""
    print("\n=== Testing Market & Macro Features ===")
    
    # Initialize DataFetcher
    fetcher = DataFetcher()
    
    # Test FX Rates
    print("\n1. Testing FX Rates...")
    fx_rates = fetcher.get_fx_rates(base='USD', symbols=['EUR', 'JPY', 'GBP'])
    print(f"FX Rates Result: {fx_rates}")
    print(f"Type: {type(fx_rates)}")
    print(f"Empty? {not fx_rates}")
    
    # Test Country Risk
    print("\n2. Testing Country Risk...")
    country_risk = fetcher.get_country_risk('AAPL')
    print(f"Country Risk Result: {country_risk}")
    print(f"Type: {type(country_risk)}")
    print(f"None? {country_risk is None}")
    
    # Test VIX
    print("\n3. Testing VIX...")
    vix = fetcher.get_vix_from_fred()
    print(f"VIX Result: {vix}")
    print(f"Type: {type(vix)}")
    print(f"None? {vix is None}")
    
    # Test GARCH Volatility
    print("\n4. Testing GARCH Volatility...")
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = pd.Series(100 + np.random.randn(len(dates)).cumsum(), index=dates)
    df = pd.DataFrame({'Close': prices})
    garch_vol = fetcher.get_garch_volatility(df)
    print(f"GARCH Volatility Result: {garch_vol}")
    print(f"Type: {type(garch_vol)}")
    print(f"None? {garch_vol is None}")
    
    # Test Commodity Prices
    print("\n5. Testing Commodity Prices...")
    commodity_prices = fetcher.get_commodity_prices()
    print(f"Commodity Prices Result: {commodity_prices}")
    print(f"Type: {type(commodity_prices)}")
    print(f"Empty? {not commodity_prices}")
    
    # Test Company Commodities
    print("\n6. Testing Company Commodities...")
    company_commodities = fetcher.get_dynamic_company_commodities_gpt('AAPL')
    print(f"Company Commodities Result: {company_commodities}")
    print(f"Type: {type(company_commodities)}")
    print(f"Empty? {not company_commodities}")
    
    # Print Environment Variables (without showing full values)
    print("\n7. Checking Environment Variables...")
    env_vars = {
        'FRED_API_KEY': bool(os.getenv('FRED_API_KEY')),
        'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY'))
    }
    print(f"Environment Variables Present: {env_vars}")

if __name__ == "__main__":
    test_market_features() 