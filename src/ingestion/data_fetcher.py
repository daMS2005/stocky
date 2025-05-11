import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
import os
from arch import arch_model
from openai import OpenAI

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
            rsi = RSIIndicator(close=close, window=14)
            df['RSI'] = rsi.rsi()
            
            # MACD
            macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = BollingerBands(close=close, window=20, window_dev=2)
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
    
    def get_fx_rates(self, base='USD', symbols=['EUR', 'JPY', 'GBP']):
        """Fetch latest FX rates from Yahoo Finance."""
        try:
            rates = {}
            for symbol in symbols:
                pair = f"{symbol}{base}=X"
                ticker = yf.Ticker(pair)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    rates[symbol] = float(hist['Close'].iloc[-1])
                    print(f"✅ Fetched {symbol} rate: {rates[symbol]}")
                else:
                    print(f"⚠️ No data available for {symbol}")
            
            if rates:
                print(f"✅ Successfully fetched FX rates: {rates}")
                return rates
            else:
                print("⚠️ No FX rates found")
                return {}
        except Exception as e:
            print(f"❌ Error fetching FX rates: {str(e)}")
            return {}
    
    def get_country_risk(self, ticker: str) -> float:
        """Get country risk score based on market data."""
        try:
            # Get country from Yahoo Finance info
            stock = yf.Ticker(ticker)
            info = stock.info
            country = info.get('country', None)
            if not country:
                print(f"⚠️ No country found for {ticker}")
                return None
            
            # For US stocks, use VIX as a proxy for country risk
            if country == 'United States':
                vix = self.get_vix_from_fred()
                if vix is not None:
                    # Normalize VIX to a 0-100 scale (assuming VIX typically ranges from 10-50)
                    risk_score = min(100, max(0, (vix - 10) * (100/40)))
                    print(f"✅ Using VIX as country risk proxy for US: {risk_score}")
                    return risk_score
                else:
                    print("⚠️ Could not fetch VIX for US risk score")
            
            # For other countries, use a default value based on country
            country_risk_map = {
                'United States': 20,  # Low risk
                'Canada': 25,
                'United Kingdom': 30,
                'Germany': 25,
                'France': 30,
                'Japan': 20,
                'China': 60,
                'India': 65,
                'Australia': 25,
                'Brazil': 70,
                'South Korea': 35,
                'Italy': 45,
                'Spain': 40,
                'Netherlands': 25,
                'Switzerland': 20,
                'Sweden': 25,
                'Russia': 80,
                'Mexico': 65,
                'Singapore': 25,
                'Hong Kong': 40,
                'South Africa': 70,
                'Turkey': 75,
                'Saudi Arabia': 55
            }
            
            risk_score = country_risk_map.get(country, 50)  # Default to medium risk
            print(f"✅ Using default risk score for {country}: {risk_score}")
            return risk_score
        except Exception as e:
            print(f"❌ Error in get_country_risk: {str(e)}")
            return None
    
    def get_vix_from_fred(self):
        """Fetch latest VIX value from FRED."""
        try:
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                print("⚠️ FRED_API_KEY not set in environment variables")
                return None
            
            url = f'https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1'
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                try:
                    vix_value = float(data['observations'][0]['value'])
                    print(f"✅ Successfully fetched VIX value: {vix_value}")
                    return vix_value
                except Exception as e:
                    print(f"❌ Error parsing VIX data: {str(e)}")
                    return None
            else:
                print(f"❌ Error fetching VIX: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error in get_vix_from_fred: {str(e)}")
            return None

    def get_garch_volatility(self, df):
        """Compute GARCH(1,1) volatility from price data using the arch package."""
        try:
            if df is None or df.empty:
                print("⚠️ No price data available for GARCH calculation")
                return None
            
            # Use a longer period for more data points
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            extended_df = self.fetch_stock_data(df.name if hasattr(df, 'name') else 'AAPL', start_date, end_date)
            
            if extended_df is None or extended_df.empty:
                print("⚠️ Could not fetch extended price data for GARCH")
                return None
            
            returns = extended_df['Close'].pct_change().dropna() * 100  # percent returns
            if len(returns) < 30:  # Need enough data for GARCH
                print("⚠️ Insufficient data points for GARCH calculation")
                return None
            
            model = arch_model(returns, vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            volatility = float(res.conditional_volatility.iloc[-1])
            print(f"✅ Successfully calculated GARCH volatility: {volatility}")
            return volatility
        except Exception as e:
            print(f"❌ Error in GARCH calculation: {str(e)}")
            return None
    
    def get_commodity_prices(self, symbols=None):
        """Fetch commodity prices from Yahoo Finance."""
        try:
            if symbols is None:
                symbols = {
                    'gold': 'GC=F',      # Gold Futures
                    'silver': 'SI=F',    # Silver Futures
                    'oil': 'CL=F',       # Crude Oil Futures
                    'copper': 'HG=F',    # Copper Futures
                    'aluminum': 'ALI=F'  # Aluminum Futures
                }
            
            prices = {}
            for name, yahoo_symbol in symbols.items():
                try:
                    ticker = yf.Ticker(yahoo_symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        prices[name] = float(hist['Close'].iloc[-1])
                        print(f"✅ Fetched {name} price: {prices[name]}")
                    else:
                        print(f"⚠️ No data available for {name}")
                        prices[name] = None
                except Exception as e:
                    print(f"❌ Error fetching {name} price: {str(e)}")
                    prices[name] = None
                
            if prices:
                print(f"✅ Successfully fetched commodity prices: {prices}")
            return prices
        except Exception as e:
            print(f"❌ Error in get_commodity_prices: {str(e)}")
            return {}

    def get_dynamic_company_commodities_gpt(self, ticker):
        """Get relevant commodities for a company using GPT-4o."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            description = info.get('longBusinessSummary') or info.get('description') or ''
            if not description:
                print(f"⚠️ No company description found for {ticker}")
                return []
            
            prompt = (
                "Given the following company description, list the commodities (e.g., gold, lithium, oil, copper, aluminum, steel, etc.) "
                "that are most relevant to this company's operations, products, or supply chain. "
                "Return ONLY a comma-separated list of commodity names, nothing else.\n\n"
                f"Description:\n{description}\n\nCommodities:"
            )
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            if not client:
                print("⚠️ OpenAI client not initialized")
                return []
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Respond with only a comma-separated list of commodities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=100
            )
            
            try:
                # Get the response text and clean it
                commodities_text = response.choices[0].message.content.strip()
                # Split by comma and clean each commodity
                commodities = [c.strip().lower() for c in commodities_text.split(',')]
                # Remove any empty strings
                commodities = [c for c in commodities if c]
                
                if commodities:
                    print(f"✅ Successfully identified commodities for {ticker}: {commodities}")
                    return commodities
                else:
                    print("⚠️ No commodities identified")
                    return []
            except Exception as e:
                print(f"❌ Error parsing GPT response for commodities: {str(e)}")
                return []
        except Exception as e:
            print(f"❌ Error in get_dynamic_company_commodities_gpt: {str(e)}")
            return []

    def get_all_features(self, ticker: str, sentiment_analyzer=None, fundamental_analyzer=None, macro_fetcher=None, alt_data_fetcher=None) -> Dict[str, Any]:
        """Fetch and process all available features for a given ticker."""
        try:
            print(f"\n=== Fetching features for {ticker} ===")
            
            # Get price data and technical indicators
            print("\n1. Fetching stock data...")
            df = self.fetch_stock_data(ticker, "2022-06-01", "2022-06-30")
            if df is None or df.empty:
                print("❌ No stock data available")
                return {
                    'error': f"No data available for {ticker}",
                    'price_data': [],
                    'technical_signals': {},
                    'sentiment': {},
                    'fundamental': {},
                    'analyst_ratings': [],
                    'alternative_data': {},
                    'last_updated': None,
                    'used_features': [],
                    'market_features': {
                        'fx_rates': {},
                        'country_risk': None,
                        'vix': None,
                        'garch_volatility': None,
                        'commodity_prices': {},
                        'company_commodities': []
                    }
                }
            print(f"✅ Stock data fetched: {len(df)} rows")
            
            print("\n2. Calculating technical indicators...")
            df = self.calculate_technical_indicators(df)
            if df is None or df.empty:
                print("❌ Failed to calculate technical indicators")
                return {
                    'error': f"Failed to calculate technical indicators for {ticker}",
                    'price_data': [],
                    'technical_signals': {},
                    'sentiment': {},
                    'fundamental': {},
                    'analyst_ratings': [],
                    'alternative_data': {},
                    'last_updated': None,
                    'used_features': [],
                    'market_features': {
                        'fx_rates': {},
                        'country_risk': None,
                        'vix': None,
                        'garch_volatility': None,
                        'commodity_prices': {},
                        'company_commodities': []
                    }
                }
            print("✅ Technical indicators calculated")
            
            # Extract technical signals
            technical_signals = {}
            for col in ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'Volume_MA', 'Volume_Ratio']:
                if col in df.columns:
                    try:
                        technical_signals[col] = float(df[col].iloc[-1])
                    except (ValueError, TypeError):
                        technical_signals[col] = None
            
            used_features = list(technical_signals.keys())
            print(f"✅ Technical signals extracted: {len(technical_signals)} indicators")

            # Sentiment analysis
            print("\n3. Fetching sentiment data...")
            sentiment = {}
            if sentiment_analyzer:
                sentiment = sentiment_analyzer.get_combined_sentiment(ticker)
                used_features += [f"sentiment_{k}" for k in sentiment.get('combined_sentiment', {}).keys()]
                print("✅ Sentiment data fetched")
            else:
                print("⚠️ No sentiment analyzer provided")

            # Fundamental analysis
            print("\n4. Fetching fundamental data...")
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
            print("✅ Fundamental data fetched")

            # Analyst ratings
            print("\n5. Fetching analyst ratings...")
            analyst_ratings = self.get_analyst_ratings(ticker)
            used_features.append('analyst_ratings')
            print("✅ Analyst ratings fetched")

            # Alternative data
            print("\n6. Fetching alternative data...")
            alt_data = alt_data_fetcher.get_alternative_data(ticker) if alt_data_fetcher else {}
            if alt_data:
                used_features += list(alt_data.keys())
                print("✅ Alternative data fetched")
            else:
                print("⚠️ No alternative data available")

            # Ensure 'date' is a column in df for price_data
            df = df.reset_index().rename(columns={df.index.name or 'index': 'date'})
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)

            # Add lagged features
            print("\n7. Adding lagged features...")
            for lag in range(1, 4):
                df[f'Close_t-{lag}'] = df['Close'].shift(lag)
                df[f'Return_t-{lag}'] = df['Close'].pct_change().shift(lag)
            print("✅ Lagged features added")

            # Market & Macro Features
            print("\n8. Fetching market & macro features...")
            
            print("\n8.1 Fetching FX rates...")
            fx_rates = self.get_fx_rates(base='USD', symbols=['EUR', 'JPY', 'GBP'])
            print(f"FX rates result: {fx_rates}")
            
            print("\n8.2 Fetching country risk...")
            country_risk = self.get_country_risk(ticker)
            print(f"Country risk result: {country_risk}")
            
            print("\n8.3 Fetching VIX...")
            vix = self.get_vix_from_fred()
            print(f"VIX result: {vix}")
            
            print("\n8.4 Calculating GARCH volatility...")
            garch_volatility = self.get_garch_volatility(df)
            print(f"GARCH volatility result: {garch_volatility}")
            
            print("\n8.5 Fetching commodity prices...")
            commodity_prices = self.get_commodity_prices()
            print(f"Commodity prices result: {commodity_prices}")
            
            print("\n8.6 Fetching company commodities...")
            company_commodities = self.get_dynamic_company_commodities_gpt(ticker)
            print(f"Company commodities result: {company_commodities}")

            # Ensure all market & macro features are included in used_features
            used_features.extend([
                'fx_rates',
                'country_risk',
                'vix',
                'garch_volatility',
                'commodity_prices',
                'company_commodities'
            ])

            # Organize market features
            market_features = {
                'fx_rates': fx_rates or {},
                'country_risk': country_risk,
                'vix': vix,
                'garch_volatility': garch_volatility,
                'commodity_prices': commodity_prices or {},
                'company_commodities': company_commodities or []
            }

            features = {
                'price_data': df.to_dict(orient='records'),
                'technical_signals': technical_signals,
                'sentiment': sentiment,
                'fundamental': fundamental,
                'analyst_ratings': analyst_ratings.to_dict(orient='records') if hasattr(analyst_ratings, 'to_dict') else analyst_ratings,
                'alternative_data': alt_data,
                'last_updated': df['date'].iloc[-1] if 'date' in df.columns else None,
                'used_features': used_features,
                'market_features': market_features  # Add market features as a separate section
            }
            
            print("\n=== Feature fetching complete ===")
            return features
            
        except Exception as e:
            print(f"\n❌ Error in get_all_features for {ticker}: {str(e)}")
            return {
                'error': f"Error processing data for {ticker}: {str(e)}",
                'price_data': [],
                'technical_signals': {},
                'sentiment': {},
                'fundamental': {},
                'analyst_ratings': [],
                'alternative_data': {},
                'last_updated': None,
                'used_features': [],
                'market_features': {
                    'fx_rates': {},
                    'country_risk': None,
                    'vix': None,
                    'garch_volatility': None,
                    'commodity_prices': {},
                    'company_commodities': []
                }
            } 