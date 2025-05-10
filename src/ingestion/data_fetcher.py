import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class DataFetcher:
    def __init__(self):
        self.cutoff_date = datetime(2022, 6, 30)
        
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df is None or df.empty:
                print(f"No data available for {ticker} between {start_date} and {end_date}")
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Convert index to timezone-naive datetime
            df.index = df.index.tz_localize(None)
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == '' or col[1] == ticker else f"{col[0]}_{col[1]}" for col in df.columns]
            return df
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {str(e)}")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from price data."""
        if df is None or df.empty:
            return df
        
        try:
            # Ensure we're working with pandas Series
            close = df['Close']
            volume = df['Volume']
            
            # RSI
            rsi = RSIIndicator(close=close)
            df['RSI'] = rsi.rsi()
            
            # MACD
            macd = MACD(close=close)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = BollingerBands(close=close)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Middle'] = bb.bollinger_mavg()
            
            # Volume analysis
            df['Volume_MA'] = volume.rolling(window=20).mean()
            df['Volume_Ratio'] = volume / df['Volume_MA']
            
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def get_financial_ratios(self, ticker: str) -> Dict[str, float]:
        """Fetch key financial ratios."""
        stock = yf.Ticker(ticker)
        info = stock.info
        
        ratios = {
            'debt_to_equity': info.get('debtToEquity', None),
            'return_on_equity': info.get('returnOnEquity', None),
            'return_on_assets': info.get('returnOnAssets', None),
            'beta': info.get('beta', None),
            'dividend_yield': info.get('dividendYield', None),
            'institutional_ownership': info.get('institutionOwnership', None),
            'short_interest_ratio': info.get('shortRatio', None)
        }
        
        return {k: v for k, v in ratios.items() if v is not None}
    
    def get_earnings_data(self, ticker: str) -> pd.DataFrame:
        """Fetch historical earnings data."""
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings
            if earnings is None:
                return pd.DataFrame()
            return earnings
        except Exception as e:
            print(f"Error fetching earnings data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_analyst_ratings(self, ticker: str) -> pd.DataFrame:
        """Fetch analyst recommendations."""
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            if recommendations is None:
                return pd.DataFrame()
            return recommendations
        except Exception as e:
            print(f"Error fetching analyst ratings for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_all_features(self, ticker: str, sentiment_analyzer=None, fundamental_analyzer=None, macro_fetcher=None, alt_data_fetcher=None) -> Dict[str, Any]:
        """Fetch and process all available features for a given ticker."""
        try:
            # Get price data and technical indicators
            df = self.fetch_stock_data(ticker, "2022-06-01", "2022-06-30")
            if df is None or df.empty:
                return {
                    'error': f"No data available for {ticker}",
                    'price_data': [],
                    'technical_signals': {},
                    'sentiment': {},
                    'fundamental': {},
                    'analyst_ratings': [],
                    'alternative_data': {},
                    'last_updated': None,
                    'used_features': []
                }
            
            df = self.calculate_technical_indicators(df)
            if df is None or df.empty:
                return {
                    'error': f"Failed to calculate technical indicators for {ticker}",
                    'price_data': [],
                    'technical_signals': {},
                    'sentiment': {},
                    'fundamental': {},
                    'analyst_ratings': [],
                    'alternative_data': {},
                    'last_updated': None,
                    'used_features': []
                }
            
            # Extract technical signals
            technical_signals = {}
            for col in ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'Volume_MA', 'Volume_Ratio']:
                if col in df.columns:
                    try:
                        technical_signals[col] = float(df[col].iloc[-1])
                    except (ValueError, TypeError):
                        technical_signals[col] = None
            
            used_features = list(technical_signals.keys())

            # Sentiment analysis
            sentiment = {}
            if sentiment_analyzer:
                sentiment = sentiment_analyzer.get_combined_sentiment(ticker)
                used_features += [f"sentiment_{k}" for k in sentiment.get('combined_sentiment', {}).keys()]

            # Fundamental analysis
            fundamental = {
                'financial_ratios': self.get_financial_ratios(ticker),
                'earnings': self.get_earnings_data(ticker).to_dict(orient='records') if not self.get_earnings_data(ticker).empty else [],
                'macro_indicators': macro_fetcher.get_macro_indicators() if macro_fetcher else {},
                'insider_activity': fundamental_analyzer.get_insider_activity(ticker) if fundamental_analyzer else {},
                'institutional_ownership': fundamental_analyzer.get_institutional_ownership(ticker) if fundamental_analyzer else {},
                'sec_filings': fundamental_analyzer.get_sec_filings(ticker) if fundamental_analyzer else {},
            }
            used_features += list(fundamental['financial_ratios'].keys())
            used_features += ['earnings', 'macro_indicators', 'insider_activity', 'institutional_ownership', 'sec_filings']

            # Analyst ratings
            analyst_ratings = self.get_analyst_ratings(ticker)
            used_features.append('analyst_ratings')

            # Alternative data
            alt_data = alt_data_fetcher.get_alternative_data(ticker) if alt_data_fetcher else {}
            if alt_data:
                used_features += list(alt_data.keys())

            # Ensure 'date' is a column in df for price_data
            df = df.reset_index().rename(columns={df.index.name or 'index': 'date'})
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)

            features = {
                'price_data': df.to_dict(orient='records'),
                'technical_signals': technical_signals,
                'sentiment': sentiment,
                'fundamental': fundamental,
                'analyst_ratings': analyst_ratings.to_dict(orient='records') if hasattr(analyst_ratings, 'to_dict') else analyst_ratings,
                'alternative_data': alt_data,
                'last_updated': df['date'].iloc[-1] if 'date' in df.columns else None,
                'used_features': used_features
            }
            return features
        except Exception as e:
            print(f"Error in get_all_features for {ticker}: {str(e)}")
            return {
                'error': f"Error processing data for {ticker}: {str(e)}",
                'price_data': [],
                'technical_signals': {},
                'sentiment': {},
                'fundamental': {},
                'analyst_ratings': [],
                'alternative_data': {},
                'last_updated': None,
                'used_features': []
            } 