import pandas as pd
import numpy as np
from typing import Dict, Any
import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

class TechnicalAnalyzer:
    def __init__(self):
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        self.volume_ma_period = 20
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given price data."""
        # RSI
        rsi = RSIIndicator(close=df['Close'], window=self.rsi_period)
        df['RSI'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=self.bb_period, window_dev=self.bb_std)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume Analysis
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_ma_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Spike'] = df['Volume_Ratio'] > 2.0  # Volume spike threshold
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        )
        df['VWAP'] = vwap.volume_weighted_average_price()
        
        # Price Momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_MA'] = df['Price_Change'].rolling(window=20).mean()
        
        # Volatility
        df['Daily_Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Trend Strength
        df['ADX'] = ta.trend.ADXIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ).adx()
        
        return df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a summary of technical signals."""
        latest = df.iloc[-1]
        
        signals = {
            'rsi_signal': 'Oversold' if latest['RSI'] < 30 else 'Overbought' if latest['RSI'] > 70 else 'Neutral',
            'macd_signal': 'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish',
            'bb_signal': 'Upper' if latest['Close'] > latest['BB_Upper'] else 'Lower' if latest['Close'] < latest['BB_Lower'] else 'Middle',
            'volume_signal': 'High' if latest['Volume_Spike'] else 'Normal',
            'trend_strength': 'Strong' if latest['ADX'] > 25 else 'Weak',
            'volatility': 'High' if latest['Daily_Volatility'] > df['Daily_Volatility'].mean() else 'Low'
        }
        
        return signals 