import unittest
from src.ingestion.data_fetcher import DataFetcher
import pandas as pd
from datetime import datetime

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.data_fetcher = DataFetcher()
        self.test_ticker = "AAPL"
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"

    def test_fetch_stock_data(self):
        """Test fetching stock data"""
        df = self.data_fetcher.fetch_stock_data(self.test_ticker, self.start_date, self.end_date)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('Close', df.columns)
        self.assertIn('Volume', df.columns)

    def test_calculate_technical_indicators(self):
        """Test technical indicator calculations"""
        df = self.data_fetcher.fetch_stock_data(self.test_ticker, self.start_date, self.end_date)
        df_with_indicators = self.data_fetcher.calculate_technical_indicators(df)
        
        # Check if technical indicators were added
        self.assertIn('RSI', df_with_indicators.columns)
        self.assertIn('MACD', df_with_indicators.columns)
        self.assertIn('BB_Upper', df_with_indicators.columns)
        self.assertIn('Volume_MA', df_with_indicators.columns)

    def test_get_financial_ratios(self):
        """Test fetching financial ratios"""
        ratios = self.data_fetcher.get_financial_ratios(self.test_ticker)
        self.assertIsInstance(ratios, dict)
        self.assertTrue(len(ratios) > 0)

    def test_get_earnings_data(self):
        """Test fetching earnings data"""
        earnings = self.data_fetcher.get_earnings_data(self.test_ticker)
        self.assertIsInstance(earnings, pd.DataFrame)

    def test_get_analyst_ratings(self):
        """Test fetching analyst ratings"""
        ratings = self.data_fetcher.get_analyst_ratings(self.test_ticker)
        self.assertIsInstance(ratings, pd.DataFrame)

    def test_get_all_features(self):
        """Test getting all features"""
        features = self.data_fetcher.get_all_features(self.test_ticker)
        self.assertIsInstance(features, dict)
        self.assertIn('price_data', features)
        self.assertIn('technical_signals', features)
        self.assertIn('fundamental', features)

if __name__ == '__main__':
    unittest.main() 