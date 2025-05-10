from src.ingestion.data_fetcher import DataFetcher

def test_get_all_features_aapl():
    fetcher = DataFetcher()
    features = fetcher.get_all_features('AAPL')
    assert isinstance(features['fx_rates'], dict), 'FX rates should be a dict'
    assert features['fx_rates'], 'FX rates should not be empty'
    assert features['country_risk'] is not None, 'Country risk should not be None'
    assert isinstance(features['commodity_prices'], dict), 'Commodity prices should be a dict'
    assert features['commodity_prices'], 'Commodity prices should not be empty'
    assert isinstance(features['company_commodities'], list), 'Company commodities should be a list'
    assert features['vix'] is not None, 'VIX should not be None'
    assert features['garch_volatility'] is not None, 'GARCH volatility should not be None'
    # Check lagged features in price_data
    last = features['price_data'][-1]
    assert 'Close_t-1' in last and 'Return_t-1' in last, 'Lagged features missing in price_data'

if __name__ == '__main__':
    test_get_all_features_aapl()
    print('All DataFetcher feature tests passed!') 