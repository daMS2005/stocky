import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SentimentVisualizer:
    def __init__(self):
        # Set the style for matplotlib plots
        plt.style.use('seaborn-v0_8')
        # Set the style for seaborn plots
        sns.set_theme(style="whitegrid")
        
    def create_sentiment_timeline(self, sentiment_history: List[Dict], ticker: str) -> go.Figure:
        """Create a timeline of sentiment scores."""
        if not sentiment_history:
            return go.Figure()
        
        df = pd.DataFrame([
            {
                'date': entry['date'],
                'score': entry.get('score', 0),
                'magnitude': entry.get('magnitude', 0)
            }
            for entry in sentiment_history
        ])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['score'],
            mode='lines+markers',
            name='Sentiment Score'
        ))
        
        fig.update_layout(
            title=f'Sentiment Timeline for {ticker}',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            showlegend=True
        )
        return fig
    
    def create_sentiment_heatmap(self, sentiment_history: List[Dict], ticker: str) -> go.Figure:
        """Create a heatmap of sentiment scores."""
        if not sentiment_history:
            return go.Figure()
        
        df = pd.DataFrame([
            {
                'date': entry['date'],
                'score': entry.get('score', 0),
                'magnitude': entry.get('magnitude', 0)
            }
            for entry in sentiment_history
        ])
        
        # Create a pivot table for the heatmap
        pivot = df.pivot_table(
            values='score',
            index=df['date'].dt.date,
            columns=df['date'].dt.hour,
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn'
        ))
        
        fig.update_layout(
            title=f'Sentiment Heatmap for {ticker}',
            xaxis_title='Hour',
            yaxis_title='Date'
        )
        return fig
    
    def create_sentiment_radar(self, sentiment_data: Dict, ticker: str) -> go.Figure:
        """Create a radar chart of sentiment components."""
        if not sentiment_data or 'combined_sentiment' not in sentiment_data:
            return go.Figure()
        
        sentiment = sentiment_data['combined_sentiment']
        categories = list(sentiment.keys())
        values = list(sentiment.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Sentiment'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-1, 1]
                )
            ),
            title=f'Sentiment Distribution for {ticker}'
        )
        return fig
    
    def create_trend_analysis(self, sentiment_history: List[Dict], ticker: str) -> go.Figure:
        """Create a trend analysis of sentiment scores."""
        if not sentiment_history:
            return go.Figure()
        
        df = pd.DataFrame([
            {
                'date': entry['date'],
                'score': entry.get('score', 0),
                'magnitude': entry.get('magnitude', 0)
            }
            for entry in sentiment_history
        ])
        
        # Calculate moving averages
        df['MA7'] = df['score'].rolling(window=7).mean()
        df['MA30'] = df['score'].rolling(window=30).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['score'],
            mode='lines',
            name='Daily Score'
        ))
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['MA7'],
            mode='lines',
            name='7-day MA'
        ))
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['MA30'],
            mode='lines',
            name='30-day MA'
        ))
        
        fig.update_layout(
            title=f'Sentiment Trend Analysis for {ticker}',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            showlegend=True
        )
        return fig 