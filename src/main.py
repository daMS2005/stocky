import os
from dotenv import load_dotenv
from typing import Dict, Any
from ingestion.data_fetcher import DataFetcher
from ingestion.technical_indicators import TechnicalAnalyzer
from ingestion.sentiment_analyzer import SentimentAnalyzer
from ingestion.fundamental_analyzer import FundamentalAnalyzer
from gpt_agent.analyst import GPTAnalyst

# Load environment variables
load_dotenv()

class StockAnalyst:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.gpt_analyst = GPTAnalyst()
    
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze a stock and provide trading recommendations.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dict[str, Any]: Trading recommendation with action, prices, and reasoning
        """
        try:
            # Fetch price data
            df = self.data_fetcher.fetch_stock_data(ticker)
            
            # Calculate technical indicators
            df = self.technical_analyzer.calculate_all_indicators(df)
            technical_signals = self.technical_analyzer.get_signal_summary(df)
            
            # Get sentiment analysis
            sentiment_data = self.sentiment_analyzer.get_combined_sentiment(ticker)
            
            # Get fundamental data
            fundamental_data = self.fundamental_analyzer.get_all_fundamental_data(ticker)
            
            # Combine all features
            features = {
                'price_data': df.to_dict(orient='records'),
                'technical_signals': technical_signals,
                'sentiment': sentiment_data,
                'fundamental': fundamental_data,
                'last_updated': df.index[-1].strftime('%Y-%m-%d')
            }
            
            # Get GPT recommendation
            recommendation = self.gpt_analyst.get_recommendation(features, ticker)
            
            return {
                'ticker': ticker,
                'analysis_date': features['last_updated'],
                'technical_analysis': technical_signals,
                'sentiment_analysis': sentiment_data['combined_sentiment'],
                'fundamental_analysis': {
                    'financial_ratios': fundamental_data['financial_ratios'],
                    'earnings': fundamental_data['earnings'],
                    'macro_indicators': fundamental_data['macro_indicators']
                },
                'recommendation': recommendation
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'error': str(e),
                'recommendation': {
                    'action': 'ERROR',
                    'entry_price': None,
                    'target_price': None,
                    'stop_loss': None,
                    'reasoning': f'Error during analysis: {str(e)}'
                }
            }

def main():
    # Example usage
    analyst = StockAnalyst()
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    
    result = analyst.analyze_stock(ticker)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Ticker: {result['ticker']}")
    print(f"Analysis Date: {result['analysis_date']}")
    
    print("\nTechnical Analysis:")
    for signal, value in result['technical_analysis'].items():
        print(f"- {signal}: {value}")
    
    print("\nSentiment Analysis:")
    sentiment = result['sentiment_analysis']
    print(f"- Positive: {sentiment['positive']:.2%}")
    print(f"- Negative: {sentiment['negative']:.2%}")
    print(f"- Neutral: {sentiment['neutral']:.2%}")
    
    print("\nFundamental Analysis:")
    ratios = result['fundamental_analysis']['financial_ratios']
    print("Financial Ratios:")
    for ratio, value in ratios.items():
        if value is not None:
            print(f"- {ratio}: {value:.2f}")
    
    print("\nMacro Indicators:")
    macro = result['fundamental_analysis']['macro_indicators']
    for indicator, value in macro.items():
        print(f"- {indicator}: {value:.2f}")
    
    print("\nTrading Recommendation:")
    print(f"Action: {result['recommendation']['action']}")
    print(f"Entry Price: ${result['recommendation']['entry_price']:.2f}")
    print(f"Target Price: ${result['recommendation']['target_price']:.2f}")
    print(f"Stop Loss: ${result['recommendation']['stop_loss']:.2f}")
    print("\nReasoning:")
    print(result['recommendation']['reasoning'])

if __name__ == "__main__":
    main() 