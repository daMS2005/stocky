import unittest
from src.ingestion.sentiment_analyzer import SentimentAnalyzer
import pandas as pd
from datetime import datetime, timedelta

class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.test_ticker = "AAPL"
        self.test_date = datetime.now()

    def test_get_combined_sentiment(self):
        """Test getting combined sentiment"""
        sentiment = self.sentiment_analyzer.get_combined_sentiment(self.test_ticker)
        self.assertIsInstance(sentiment, dict)
        self.assertIn('combined_sentiment', sentiment)

    def test_get_sentiment_history(self):
        """Test getting sentiment history"""
        history = self.sentiment_analyzer.sentiment_history.get(self.test_ticker, [])
        self.assertIsInstance(history, list)

    def test_analyze_sentiment(self):
        """Test sentiment analysis"""
        # Test with a sample text
        sample_text = "Apple's new iPhone shows strong sales and positive market reception."
        sentiment = self.sentiment_analyzer.analyze_sentiment(sample_text)
        self.assertIsInstance(sentiment, dict)
        self.assertIn('score', sentiment)
        self.assertIn('magnitude', sentiment)

    def test_update_sentiment_history(self):
        """Test updating sentiment history"""
        initial_length = len(self.sentiment_analyzer.sentiment_history.get(self.test_ticker, []))
        
        # Add a new sentiment entry
        new_sentiment = {
            'date': self.test_date,
            'score': 0.8,
            'magnitude': 0.9,
            'source': 'test'
        }
        self.sentiment_analyzer.sentiment_history.setdefault(self.test_ticker, []).append(new_sentiment)
        
        # Check if the history was updated
        updated_length = len(self.sentiment_analyzer.sentiment_history.get(self.test_ticker, []))
        self.assertEqual(updated_length, initial_length + 1)

if __name__ == '__main__':
    unittest.main() 