import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from src.rag_db import get_vector_store
from src.prediction_logger import PredictionLogger
from src.ingestion.data_fetcher import DataFetcher
import openai
import os
from src.scripts.run_learning_phase import build_prediction_prompt, adjust_prediction, get_previous_prediction

def load_metrics():
    """Load learning metrics from JSON file."""
    try:
        with open("logs/learning_metrics.json", 'r') as f:
            return json.load(f)
    except:
        return []

def load_test_results():
    """Load test results from JSON file."""
    try:
        with open("logs/test_results.json", 'r') as f:
            return json.load(f)
    except:
        return {}

def plot_learning_metrics(metrics):
    """Plot learning metrics over time."""
    if not metrics:
        return None
    
    df = pd.DataFrame(metrics)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy Over Time', 'Average Return Over Time'))
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['overall_accuracy'], name='Overall Accuracy'),
        row=1, col=1
    )
    
    if 'rolling_accuracy' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['rolling_accuracy'], name='Rolling Accuracy'),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['average_return'], name='Average Return'),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="Learning Phase Metrics")
    return fig

def plot_prediction_breakdown(test_results):
    """Plot prediction breakdown and returns."""
    if not test_results:
        return None
    
    # Prediction breakdown
    breakdown = test_results.get('prediction_breakdown', {})
    returns = test_results.get('return_by_prediction', {})
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Prediction Distribution', 'Average Returns by Prediction'))
    
    fig.add_trace(
        go.Bar(x=list(breakdown.keys()), y=list(breakdown.values()), name='Count'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=list(returns.keys()), y=list(returns.values()), name='Return'),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="Test Phase Results")
    return fig

def plot_pnl_metrics(test_results):
    """Plot PnL metrics."""
    if not test_results or 'pnl_metrics' not in test_results:
        return None
    
    pnl = test_results['pnl_metrics']
    
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=('Win Rate', 'Average Win/Loss', 
                                     'Largest Win/Loss', 'Sharpe Ratio'))
    
    # Win Rate
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=pnl['win_rate'] * 100,
            title={'text': "Win Rate (%)"},
            gauge={'axis': {'range': [0, 100]}},
        ),
        row=1, col=1
    )
    
    # Average Win/Loss
    fig.add_trace(
        go.Bar(
            x=['Win', 'Loss'],
            y=[pnl['average_win'], pnl['average_loss']],
            name='Average'
        ),
        row=1, col=2
    )
    
    # Largest Win/Loss
    fig.add_trace(
        go.Bar(
            x=['Win', 'Loss'],
            y=[pnl['largest_win'], pnl['largest_loss']],
            name='Largest'
        ),
        row=2, col=1
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=pnl['sharpe_ratio'],
            title={'text': "Sharpe Ratio"},
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="PnL Metrics")
    return fig

def get_current_prediction(ticker: str):
    """Get current prediction and analysis."""
    try:
        # Initialize components
        vector_store = get_vector_store('faiss', embedding_dim=1536)
        logger = PredictionLogger()
        data_fetcher = DataFetcher()
        
        # Get current date
        current_date = datetime.now().date()
        week_start = current_date - timedelta(days=current_date.weekday())
        week_end = week_start + timedelta(days=4)
        
        # Fetch current data
        df = data_fetcher.fetch_stock_data(ticker, 
                                         (week_start - timedelta(days=60)).strftime('%Y-%m-%d'),
                                         week_end.strftime('%Y-%m-%d'))
        
        if df.empty:
            return None, None, None, None, None, None
        
        # Calculate features
        df = data_fetcher.calculate_technical_indicators(df)
        week_df = df[df.index.date >= week_start]
        
        # Get current features
        current_features = {
            'technical': {
                'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 0.0,
                'macd': float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else 0.0,
                'bollinger_upper': float(df['BB_Upper'].iloc[-1]) if 'BB_Upper' in df.columns and not pd.isna(df['BB_Upper'].iloc[-1]) else 0.0,
                'bollinger_lower': float(df['BB_Lower'].iloc[-1]) if 'BB_Lower' in df.columns and not pd.isna(df['BB_Lower'].iloc[-1]) else 0.0,
                'bollinger_middle': float(df['BB_Middle'].iloc[-1]) if 'BB_Middle' in df.columns and not pd.isna(df['BB_Middle'].iloc[-1]) else 0.0,
                'volume_ratio': float(df['Volume_Ratio'].iloc[-1]) if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]) else 0.0
            },
            'price_data': {
                'open_price': float(week_df['Open'].iloc[0]),
                'close_price': float(week_df['Close'].iloc[-1]),
                'weekly_return': float(((week_df['Close'].iloc[-1] - week_df['Open'].iloc[0]) / week_df['Open'].iloc[0]) * 100)
            }
        }
        
        # Get fundamental and macro data
        try:
            ratios = data_fetcher.get_financial_ratios(ticker)
            if isinstance(ratios, dict):
                current_features['fundamental'] = {
                    'debt_to_equity': float(ratios.get('debt_to_equity', 0.0)),
                    'return_on_equity': float(ratios.get('return_on_equity', 0.0)),
                    'beta': float(ratios.get('beta', 0.0))
                }
        except:
            current_features['fundamental'] = {
                'debt_to_equity': 0.0,
                'return_on_equity': 0.0,
                'beta': 0.0
            }
        
        try:
            country_risk = data_fetcher.get_country_risk(ticker)
            vix = data_fetcher.get_vix_from_fred()
            
            current_features['macro'] = {
                'country_risk': float(country_risk) if country_risk is not None else 0.0,
                'vix': float(vix) if vix is not None else 0.0
            }
        except:
            current_features['macro'] = {
                'country_risk': 0.0,
                'vix': 0.0
            }
        
        # Get similar past weeks
        feature_values = []
        for category in ['technical', 'price_data', 'fundamental', 'macro']:
            if category in current_features:
                for value in current_features[category].values():
                    if isinstance(value, (int, float)):
                        feature_values.append(float(value))
                    elif isinstance(value, dict):
                        for v in value.values():
                            if isinstance(v, (int, float)):
                                feature_values.append(float(v))
        
        # Pad or truncate to match embedding dimension
        if len(feature_values) < 1536:
            feature_values.extend([0.0] * (1536 - len(feature_values)))
        else:
            feature_values = feature_values[:1536]
        
        similar_weeks = vector_store.query_similar(feature_values, k=5)
        
        # Get previous prediction
        previous_prediction = get_previous_prediction(logger, ticker, current_date)
        
        # Make prediction
        prompt = build_prediction_prompt(ticker, current_features, similar_weeks)
        
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a performance-aware stock analyst. Learn from past predictions and adapt your strategy. Provide concise, well-formatted responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        
        # Extract prediction components
        prediction = None
        confidence = None
        reasoning = None
        features = []
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('PREDICTION:'):
                prediction = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
            elif line.startswith('FEATURES:'):
                features = [f.strip() for f in line.split(':', 1)[1].strip().split(',')]
        
        # Adjust prediction based on trend continuation
        if prediction and previous_prediction:
            adjusted_prediction = adjust_prediction(prediction, previous_prediction)
            if adjusted_prediction != prediction:
                prediction = adjusted_prediction
        
        return prediction, confidence, reasoning, features, current_features, similar_weeks
        
    except Exception as e:
        st.error(f"Error getting prediction: {str(e)}")
        return None, None, None, None, None, None

def plot_technical_indicators(df):
    """Plot technical indicators."""
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Price and Bollinger Bands', 'RSI', 'MACD'))
    
    # Price and Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle', line=dict(color='gray')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray')),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
            row=3, col=1
        )
        if 'MACD_Signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'),
                row=3, col=1
            )
    
    fig.update_layout(height=1000, title_text="Technical Analysis")
    return fig

def main():
    st.set_page_config(page_title="Stock Predictor", layout="wide")
    
    st.title("Stock Predictor Dashboard")
    
    # Sidebar
    st.sidebar.title("Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    
    # Main content
    tabs = st.tabs(["Current Analysis", "Learning Metrics", "Test Results", "Technical Analysis"])
    
    # Current Analysis Tab
    with tabs[0]:
        st.header("Current Market Analysis")
        
        if st.button("Get Current Prediction"):
            with st.spinner("Analyzing market..."):
                prediction, confidence, reasoning, features, current_features, similar_weeks = get_current_prediction(ticker)
                
                if prediction:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction")
                        st.metric("Action", prediction, f"{confidence:.0%} confidence")
                        
                        st.subheader("Reasoning")
                        st.write(reasoning)
                        
                        st.subheader("Key Features")
                        for feature in features:
                            st.write(f"- {feature}")
                    
                    with col2:
                        st.subheader("Current Market State")
                        if current_features:
                            st.write("Technical Indicators:")
                            for k, v in current_features['technical'].items():
                                st.write(f"- {k}: {v:.2f}")
                            
                            st.write("\nPrice Data:")
                            for k, v in current_features['price_data'].items():
                                st.write(f"- {k}: {v:.2f}")
                            
                            if 'fundamental' in current_features:
                                st.write("\nFundamental Data:")
                                for k, v in current_features['fundamental'].items():
                                    st.write(f"- {k}: {v:.2f}")
                            
                            if 'macro' in current_features:
                                st.write("\nMacro Data:")
                                for k, v in current_features['macro'].items():
                                    st.write(f"- {k}: {v:.2f}")
                    
                    st.subheader("Similar Past Weeks")
                    if similar_weeks:
                        for week in similar_weeks:
                            st.write(f"Week of {week.get('week_start', 'N/A')}: {week.get('outcome', 'N/A')} → {week.get('price_data', {}).get('weekly_return', 0.0):.2f}%")
                else:
                    st.error("Could not get prediction. Please check the ticker symbol and try again.")
    
    # Learning Metrics Tab
    with tabs[1]:
        st.header("Learning Phase Metrics")
        metrics = load_metrics()
        if metrics:
            fig = plot_learning_metrics(metrics)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No learning metrics available yet.")
    
    # Test Results Tab
    with tabs[2]:
        st.header("Test Phase Results")
        test_results = load_test_results()
        if test_results:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_prediction_breakdown(test_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plot_pnl_metrics(test_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No test results available yet.")
    
    # Technical Analysis Tab
    with tabs[3]:
        st.header("Technical Analysis")
        if st.button("Update Technical Analysis"):
            with st.spinner("Fetching technical data..."):
                data_fetcher = DataFetcher()
                df = data_fetcher.fetch_stock_data(ticker, 
                                                 (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
                                                 datetime.now().strftime('%Y-%m-%d'))
                
                if not df.empty:
                    df = data_fetcher.calculate_technical_indicators(df)
                    fig = plot_technical_indicators(df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not fetch technical data. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main() 