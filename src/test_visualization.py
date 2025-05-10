import unittest
from src.visualization.sentiment_visualizer import SentimentVisualizer
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

class TestSentimentVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = SentimentVisualizer()
        self.test_ticker = "AAPL"
        self.test_sentiment_history = [
            {
                'date': datetime.now() - timedelta(days=i),
                'score': 0.8,
                'magnitude': 0.9,
                'source': 'test'
            }
            for i in range(5)
        ]

    def test_create_sentiment_timeline(self):
        """Test creating sentiment timeline"""
        fig = self.visualizer.create_sentiment_timeline(
            self.test_sentiment_history,
            self.test_ticker
        )
        self.assertIsInstance(fig, go.Figure)

    def test_create_sentiment_heatmap(self):
        """Test creating sentiment heatmap"""
        fig = self.visualizer.create_sentiment_heatmap(
            self.test_sentiment_history,
            self.test_ticker
        )
        self.assertIsInstance(fig, go.Figure)

    def test_create_sentiment_radar(self):
        """Test creating sentiment radar chart"""
        sentiment_data = {
            'combined_sentiment': {
                'overall': 0.8,
                'news': 0.7,
                'social': 0.9,
                'technical': 0.6
            }
        }
        fig = self.visualizer.create_sentiment_radar(sentiment_data, self.test_ticker)
        self.assertIsInstance(fig, go.Figure)

    def test_create_trend_analysis(self):
        """Test creating trend analysis"""
        fig = self.visualizer.create_trend_analysis(
            self.test_sentiment_history,
            self.test_ticker
        )
        self.assertIsInstance(fig, go.Figure)

if __name__ == '__main__':
    unittest.main() 