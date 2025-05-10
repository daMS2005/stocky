import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import yfinance as yf

class FundamentalAnalyzer:
    def __init__(self):
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
    def get_financial_ratios(self, ticker: str) -> Dict[str, float]:
        """Get comprehensive financial ratios from FMP."""
        url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={self.fmp_api_key}'
        response = requests.get(url)
        
        if response.status_code == 200:
            ratios = response.json()[0]  # Get most recent ratios
            return {
                'debt_to_equity': ratios.get('debtToEquity', None),
                'return_on_equity': ratios.get('returnOnEquity', None),
                'return_on_assets': ratios.get('returnOnAssets', None),
                'profit_margin': ratios.get('profitMargin', None),
                'operating_margin': ratios.get('operatingMargin', None),
                'current_ratio': ratios.get('currentRatio', None),
                'quick_ratio': ratios.get('quickRatio', None),
                'interest_coverage': ratios.get('interestCoverage', None),
                'dividend_yield': ratios.get('dividendYield', None),
                'payout_ratio': ratios.get('payoutRatio', None)
            }
        return {}
    
    def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data and surprises."""
        url = f'https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={self.fmp_api_key}'
        response = requests.get(url)
        
        if response.status_code == 200:
            earnings = response.json()
            return {
                'recent_earnings': earnings[:4],  # Last 4 quarters
                'avg_surprise': np.mean([e.get('surprisePercentage', 0) for e in earnings[:4]]),
                'surprise_volatility': np.std([e.get('surprisePercentage', 0) for e in earnings[:4]])
            }
        return {}
    
    def get_insider_trading(self, ticker: str) -> List[Dict[str, Any]]:
        """Get recent insider trading activity."""
        url = f'https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&apikey={self.fmp_api_key}'
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()[:10]  # Last 10 insider trades
        return []
    
    def get_institutional_ownership(self, ticker: str) -> Dict[str, Any]:
        """Get institutional ownership trends."""
        url = f'https://financialmodelingprep.com/api/v3/institutional-holder/{ticker}?apikey={self.fmp_api_key}'
        response = requests.get(url)
        
        if response.status_code == 200:
            holders = response.json()
            return {
                'total_shares': sum(h.get('shares', 0) for h in holders),
                'holder_count': len(holders),
                'top_holders': holders[:5]  # Top 5 institutional holders
            }
        return {}
    
    def get_sec_filings(self, ticker: str) -> List[Dict[str, Any]]:
        """Get recent SEC filings."""
        url = f'https://financialmodelingprep.com/api/v3/sec_filings/{ticker}?type=10-K,10-Q,8-K&apikey={self.fmp_api_key}'
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()[:5]  # Last 5 filings
        return []
    
    def get_macro_indicators(self) -> Dict[str, float]:
        """Get macroeconomic indicators from FRED."""
        indicators = {
            'FEDFUNDS': 'interest_rate',  # Federal Funds Rate
            'CPIAUCSL': 'inflation',      # CPI
            'UNRATE': 'unemployment',     # Unemployment Rate
            'GDP': 'gdp',                 # GDP
            'M2': 'money_supply'          # Money Supply
        }
        
        macro_data = {}
        for fred_id, indicator in indicators.items():
            url = f'https://api.stlouisfed.org/fred/series/observations?series_id={fred_id}&api_key={self.fred_api_key}&file_type=json'
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data:
                    latest = data['observations'][-1]
                    macro_data[indicator] = float(latest['value'])
        
        return macro_data
    
    def get_all_fundamental_data(self, ticker: str) -> Dict[str, Any]:
        """Get all fundamental data for a ticker."""
        return {
            'financial_ratios': self.get_financial_ratios(ticker),
            'earnings': self.get_earnings_data(ticker),
            'insider_trading': self.get_insider_trading(ticker),
            'institutional_ownership': self.get_institutional_ownership(ticker),
            'sec_filings': self.get_sec_filings(ticker),
            'macro_indicators': self.get_macro_indicators()
        }

    def get_insider_activity(self, ticker: str) -> dict:
        """Stub for compatibility: returns recent insider trading activity as a dict."""
        # You can implement this to return a summary or just return an empty dict for now
        return {}

    def get_institutional_ownership(self, ticker: str) -> dict:
        """Stub for compatibility: returns institutional ownership as a dict."""
        return {}

    def get_sec_filings(self, ticker: str) -> dict:
        """Stub for compatibility: returns SEC filings as a dict."""
        return {} 