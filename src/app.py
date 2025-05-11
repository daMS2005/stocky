import os
import sys
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from src.ingestion.sentiment_analyzer import SentimentAnalyzer
from src.ingestion.fundamental_analyzer import FundamentalAnalyzer
from src.ingestion.data_fetcher import DataFetcher
from src.visualization.sentiment_visualizer import SentimentVisualizer
from src.gpt_agent.meta_agent import MetaAgentController
from src.rag_db import get_vector_store
from src.prediction_logger import PredictionLogger
from src.scripts.run_learning_phase import build_prediction_prompt, adjust_prediction, get_previous_prediction
from dotenv import load_dotenv
from src.scripts.populate_rag_db import week_key, get_existing_record, features_changed, populate_rag_db
import importlib

# Load environment variables
load_dotenv()

# Initialize components
sentiment_analyzer = SentimentAnalyzer()
fundamental_analyzer = FundamentalAnalyzer()
data_fetcher = DataFetcher()
visualizer = SentimentVisualizer()
meta_agent = MetaAgentController()

def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Uncaught exception:", exc_type, exc_value)
    traceback.print_tb(exc_traceback)

sys.excepthook = log_exception

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
    if not isinstance(test_results, dict):
        st.error("Test results are not a dictionary. Cannot plot breakdown.")
        return None
    if not test_results:
        return None
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
    # Use .get() with defaults and handle None values
    win_rate = pnl.get('win_rate', 0) or 0
    average_win = pnl.get('average_win', 0) or 0
    average_loss = pnl.get('average_loss', 0) or 0
    largest_win = pnl.get('largest_win', 0) or 0
    largest_loss = pnl.get('largest_loss', 0) or 0
    sharpe_ratio = pnl.get('sharpe_ratio', 0) or 0
    
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=('Win Rate', 'Average Win/Loss', 
                                     'Largest Win/Loss', 'Sharpe Ratio'))
    
    # Win Rate
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=win_rate * 100,
            title={'text': "Win Rate (%)"},
            gauge={'axis': {'range': [0, 100]}},
        ),
        row=1, col=1
    )
    
    # Average Win/Loss
    fig.add_trace(
        go.Bar(
            x=['Win', 'Loss'],
            y=[average_win, average_loss],
            name='Average'
        ),
        row=1, col=2
    )
    
    # Largest Win/Loss
    fig.add_trace(
        go.Bar(
            x=['Win', 'Loss'],
            y=[largest_win, largest_loss],
            name='Largest'
        ),
        row=2, col=1
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=sharpe_ratio,
            title={'text': "Sharpe Ratio"},
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="PnL Metrics")
    return fig

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

def sync_and_get_rag_data(ticker):
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    all_records = [r for r in vector_store.get_all() if r['ticker'] == ticker]
    today = datetime.now().date()
    last_monday = today - timedelta(days=today.weekday())
    new_data_added = False
    try:
        if all_records:
            latest_week = max(r['week_start'] for r in all_records)
            # If not up to last week, populate missing weeks
            if latest_week < last_monday - timedelta(days=7):
                with st.spinner(f"Updating RAG DB for {ticker} from {latest_week + timedelta(days=7)} to {last_monday - timedelta(days=7)}..."):
                    before_count = len(all_records)
                    populate_rag_db(
                        ticker,
                        (latest_week + timedelta(days=7)).strftime('%Y-%m-%d'),
                        (last_monday - timedelta(days=7)).strftime('%Y-%m-%d'),
                        force_update=False
                    )
                    all_records = [r for r in vector_store.get_all() if r['ticker'] == ticker]
                    after_count = len(all_records)
                    new_weeks = after_count - before_count
                    if new_weeks > 0:
                        st.success(f"RAG DB updated: {new_weeks} new week(s) added for {ticker}.")
                    else:
                        st.info(f"RAG DB checked: No new data needed for {ticker}.")
        else:
            # No DB for ticker, populate from earliest reasonable date
            with st.spinner(f"Populating RAG DB for {ticker} from 2010-01-01 to {last_monday - timedelta(days=7)}..."):
                populate_rag_db(
                    ticker,
                    '2010-01-01',
                    (last_monday - timedelta(days=7)).strftime('%Y-%m-%d'),
                    force_update=False
                )
                all_records = [r for r in vector_store.get_all() if r['ticker'] == ticker]
                st.success(f"RAG DB created and populated for {ticker}.")
    except Exception as e:
        st.error(f"Error updating RAG DB for {ticker}: {str(e)}")
    if not all_records:
        st.warning("No RAG data found for this ticker. Please check your data source.")
        return []
    return all_records

def display_training_report(records):
    if not records:
        st.info("No training data available.")
        return
    weeks = sorted([r['week_start'] for r in records])
    st.markdown(f"**Training period:** {weeks[0]} to {weeks[-1]}")
    st.markdown(f"**Number of weeks:** {len(weeks)}")
    st.markdown(f"**Feature keys:** {list(records[0]['features'].keys())}")

def plot_training_chart(records):
    if not records:
        return
    # Only include records with 'price_data' and 'features' as dicts
    filtered = [
        r for r in records
        if isinstance(r.get('price_data', None), dict) and isinstance(r.get('features', None), dict)
    ]
    if not filtered:
        st.info("No valid price data available in RAG records.")
        return
    df = pd.DataFrame([{
        'week_start': r['week_start'],
        'close_price': r['price_data'].get('close_price', None),
        'rsi': r['features'].get('technical', {}).get('rsi', None)
    } for r in filtered])
    df['week_start'] = pd.to_datetime(df['week_start'])
    # Data validation before plotting
    if df.empty or df['week_start'].isnull().all() or df['close_price'].isnull().all():
        st.info("No valid data to plot.")
        return
    st.line_chart(df.set_index('week_start')[['close_price', 'rsi']])

def download_features_json(records):
    if not records:
        return
    features_json = json.dumps([r['features'] for r in records], indent=2, default=str)
    st.download_button("Download Features JSON", features_json, file_name="features.json")

def main():
    # Set page config
    st.set_page_config(
        page_title="Stock Financial AI Analyst",
        page_icon="📈",
        layout="wide"
    )
    
    # Title and description
    st.title("📈 Stock Financial AI Analyst")
    st.markdown("""
    This tool analyzes a stock using technical, sentiment, fundamental, macro, and alternative data to provide a full investment report and trading recommendation.
    """)
    
    # Sidebar
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    
    # Calculate dates
    current_date = datetime.now().date()
    training_end_date = current_date - timedelta(days=7)  # End training a week ago
    prediction_start = current_date  # Start predictions from today
    prediction_end = current_date + timedelta(days=7)  # Predict for next week
    
    # Set default dates
    start_date = st.sidebar.date_input("Start Date", prediction_start)
    end_date = st.sidebar.date_input("End Date", prediction_end)
    timeframe = {"start": str(start_date), "end": str(end_date)}
    
    # Add date info
    st.sidebar.markdown(f"""
    ### Training and Prediction Periods
    - Training Data: Up to {training_end_date.strftime('%Y-%m-%d')}
    - Current Date: {current_date.strftime('%Y-%m-%d')}
    - Prediction Period: {prediction_start.strftime('%Y-%m-%d')} to {prediction_end.strftime('%Y-%m-%d')}
    """)
    
    # Main content
    if ticker:
        try:
            # Fetch features for all tabs
            with st.spinner(f"Fetching data for {ticker}..."):
                # Get data for training (up until a week ago)
                training_start = (training_end_date - timedelta(days=60)).strftime('%Y-%m-%d')
                training_end = training_end_date.strftime('%Y-%m-%d')
                
                # Get current data for prediction
                prediction_start_str = prediction_start.strftime('%Y-%m-%d')
                prediction_end_str = prediction_end.strftime('%Y-%m-%d')
                
                # Fetch training data
                training_features = data_fetcher.get_all_features(
                    ticker,
                    sentiment_analyzer=sentiment_analyzer,
                    fundamental_analyzer=fundamental_analyzer
                )
                
                # Fetch current data for prediction
                current_features = data_fetcher.get_all_features(
                    ticker,
                    sentiment_analyzer=sentiment_analyzer,
                    fundamental_analyzer=fundamental_analyzer
                )
                
                if 'error' in current_features:
                    st.error(current_features['error'])
                    st.info("Please check if the ticker is valid and try again.")
                    st.stop()
                
                if not current_features.get('price_data'):
                    st.error(f"No price data available for {ticker}")
                    st.info("Please try a different ticker or date range.")
                    st.stop()
                
                # Verify we have current data
                latest_date = None
                if isinstance(current_features.get('price_data'), dict) and 'date' in current_features['price_data']:
                    latest_date = datetime.strptime(current_features['price_data']['date'], '%Y-%m-%d').date()
                elif isinstance(current_features.get('price_data'), list) and current_features['price_data']:
                    latest_date = datetime.strptime(current_features['price_data'][-1]['date'], '%Y-%m-%d').date()
                
                if latest_date and (current_date - latest_date).days > 1:
                    st.warning(f"⚠️ Warning: Latest data is from {latest_date.strftime('%Y-%m-%d')}. Market may be closed or data may be delayed.")
            
            # --- RAG DB Sync & Training Data Visualization ---
            rag_records = sync_and_get_rag_data(ticker)

            # Tabs
            tabs = st.tabs([
                "Investment Report", 
                "Learning Metrics", 
                "Test Results", 
                "Technical Analysis",
                "Sentiment Analysis",
                "RAG Training Data"
            ])
            
            # Investment Report Tab
            with tabs[0]:
                st.header("Current Market Analysis")
                
                if st.button("Get Current Prediction"):
                    with st.spinner(f"Analyzing {ticker}..."):
                        # Get latest RAG DB record for date consistency
                        vector_store = get_vector_store('faiss', embedding_dim=1536)
                        all_records = [r for r in vector_store.get_all() if r['ticker'] == ticker]
                        if not all_records:
                            st.error(f"No RAG data found for {ticker}.")
                            st.stop()
                        
                        latest_record = max(all_records, key=lambda x: x['week_start'])
                        latest_date = latest_record['week_start']
                        
                        # Update timeframe to use RAG DB date
                        timeframe = {
                            "start": latest_date.strftime('%Y-%m-%d'),
                            "end": (latest_date + timedelta(days=7)).strftime('%Y-%m-%d')
                        }
                        
                        # Ensure logger is initialized before this call
                        logger = PredictionLogger()
                        similar_weeks = get_previous_prediction(logger, ticker, latest_date)
                        
                        # Build prediction prompt with current data
                        prompt = build_prediction_prompt(ticker, current_features, similar_weeks)
                        
                        # Run meta-agent with current data and RAG
                        result = meta_agent.make_prediction(
                            ticker, 
                            timeframe, 
                            current_features,
                            generated_on=latest_date.strftime('%B %d, %Y')  # Pass RAG DB date
                        )
                        structured = result["structured"]
                        report = result["report"]
                        
                        # Adjust prediction based on trend continuation
                        if structured and "prediction" in structured:
                            previous_prediction = get_previous_prediction(logger, ticker, latest_date)
                            adjusted_prediction = adjust_prediction(structured["prediction"], previous_prediction)
                            structured["prediction"] = adjusted_prediction
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Natural Language Report")
                            st.markdown(report)
                            
                            st.subheader("Structured Analysis")
                            st.json(structured)
                        
                        with col2:
                            st.subheader("Current Market Analysis")
                            st.write(f"Analysis Date: {latest_date.strftime('%Y-%m-%d')}")
                            st.write(f"Prediction Period: {timeframe['start']} to {timeframe['end']}")
                            
                            if similar_weeks:
                                st.write(f"Previous Prediction: {similar_weeks}")
                            
                            st.subheader("Current Market State")
                            if current_features:
                                # Technical Analysis
                                st.write("📊 Technical Indicators")
                                if isinstance(current_features.get('technical'), dict):
                                    tech_indicators = {
                                        'RSI': 'Relative Strength Index',
                                        'MACD': 'Moving Average Convergence Divergence',
                                        'BB_Upper': 'Bollinger Bands Upper',
                                        'BB_Lower': 'Bollinger Bands Lower',
                                        'BB_Middle': 'Bollinger Bands Middle',
                                        'Volume_MA': 'Volume Moving Average',
                                        'Volume_Ratio': 'Volume Ratio'
                                    }
                                    for k, v in current_features.get('technical', {}).items():
                                        if k in tech_indicators:
                                            if isinstance(v, (int, float)):
                                                st.write(f"- {tech_indicators[k]}: {v:.2f}")
                                            else:
                                                st.write(f"- {tech_indicators[k]}: {v}")
                                
                                # Sentiment Analysis
                                st.write("\n🧠 Sentiment Analysis & NLP")
                                if 'sentiment' in current_features:
                                    sentiment_data = current_features['sentiment']
                                    if isinstance(sentiment_data, dict):
                                        st.write("FinBERT/LLM Analysis:")
                                        for k, v in sentiment_data.items():
                                            if isinstance(v, (int, float)):
                                                st.write(f"- {k}: {v:.2f}")
                                            else:
                                                st.write(f"- {k}: {v}")
                                    
                                    if 'events' in sentiment_data:
                                        st.write("\nEvent Detection:")
                                        for event in sentiment_data['events']:
                                            st.write(f"- {event}")
                                
                                # Fundamental Data
                                st.write("\n💰 Fundamental & Financial Data")
                                if 'fundamental' in current_features:
                                    fundamental_data = {
                                        'eps': 'Earnings Per Share',
                                        'pe_ratio': 'P/E Ratio',
                                        'debt_to_equity': 'Debt to Equity',
                                        'roe': 'Return on Equity',
                                        'roa': 'Return on Assets',
                                        'dividend_yield': 'Dividend Yield',
                                        'insider_trading': 'Insider Trading Activity',
                                        'institutional_ownership': 'Institutional Ownership'
                                    }
                                    if isinstance(current_features['fundamental'], dict):
                                        for k, v in current_features['fundamental'].items():
                                            if k in fundamental_data:
                                                if isinstance(v, (int, float)):
                                                    st.write(f"- {fundamental_data[k]}: {v:.2f}")
                                                else:
                                                    st.write(f"- {fundamental_data[k]}: {v}")
                                
                                # News & Sentiment Features
                                st.write("\n📰 News & Sentiment Features")
                                if 'news_sentiment' in current_features:
                                    news_data = current_features['news_sentiment']
                                    if isinstance(news_data, dict):
                                        st.write("Sentiment Trends:")
                                        for k, v in news_data.items():
                                            if isinstance(v, (int, float)):
                                                st.write(f"- {k}: {v:.2f}")
                                            else:
                                                st.write(f"- {k}: {v}")
                                    
                                    if 'topics' in news_data:
                                        st.write("\nTopic Clustering:")
                                        for topic in news_data['topics']:
                                            st.write(f"- {topic}")
                                
                                # Market Features
                                st.write("\n📈 Market Features")
                                if 'market_features' in current_features:
                                    market_data = current_features['market_features']
                                    
                                    # Sector/Index Correlation
                                    if 'correlations' in market_data:
                                        st.write("Sector/Index Correlations:")
                                        for index, corr in market_data['correlations'].items():
                                            if isinstance(corr, (int, float)):
                                                st.write(f"- {index}: {corr:.2f}")
                                            else:
                                                st.write(f"- {index}: {corr}")
                                    
                                    # Beta and Short Interest
                                    if 'beta' in market_data:
                                        if isinstance(market_data['beta'], (int, float)):
                                            st.write(f"\nBeta: {market_data['beta']:.2f}")
                                        else:
                                            st.write(f"\nBeta: {market_data['beta']}")
                                    if 'short_interest' in market_data:
                                        if isinstance(market_data['short_interest'], (int, float)):
                                            st.write(f"Short Interest Ratio: {market_data['short_interest']:.2f}")
                                        else:
                                            st.write(f"Short Interest Ratio: {market_data['short_interest']}")
                                    
                                    # Analyst Ratings
                                    if 'analyst_ratings' in market_data:
                                        st.write("\nAnalyst Ratings:")
                                        for rating, count in market_data['analyst_ratings'].items():
                                            st.write(f"- {rating}: {count}")
                                
                                # Macro & Geographic Features
                                st.write("\n🌍 Macro & Geographic Features")
                                if 'macro' in current_features:
                                    macro_data = current_features['macro']
                                    if isinstance(macro_data, dict):
                                        st.write("Economic Indicators:")
                                        for k, v in macro_data.items():
                                            if isinstance(v, (int, float)):
                                                st.write(f"- {k}: {v:.2f}")
                                            else:
                                                st.write(f"- {k}: {v}")
                                    
                                    # FX Rates
                                    if 'fx_rates' in market_data:
                                        st.write("\nFX Rates:")
                                        for currency, rate in market_data['fx_rates'].items():
                                            if isinstance(rate, (int, float)):
                                                st.write(f"- {currency}: {rate:.4f}")
                                            else:
                                                st.write(f"- {currency}: {rate}")
                                    
                                    # Commodity Prices
                                    if 'commodity_prices' in market_data:
                                        st.write("\nCommodity Prices:")
                                        for commodity, price in market_data['commodity_prices'].items():
                                            if isinstance(price, (int, float)):
                                                st.write(f"- {commodity}: ${price:.2f}")
                                            else:
                                                st.write(f"- {commodity}: {price}")
                                    
                                    # Country Risk
                                    if 'country_risk' in market_data:
                                        if isinstance(market_data['country_risk'], (int, float)):
                                            st.write(f"\nCountry Risk Index: {market_data['country_risk']:.2f}")
                                        else:
                                            st.write(f"\nCountry Risk Index: {market_data['country_risk']}")
                                
                                # Behavioral & Alternative Data
                                st.write("\n🤖 Behavioral & Alternative Data")
                                if 'alternative_data' in current_features:
                                    alt_data = current_features['alternative_data']
                                    if isinstance(alt_data, dict):
                                        st.write("Social Media & Trends:")
                                        for k, v in alt_data.items():
                                            if isinstance(v, (int, float)):
                                                st.write(f"- {k}: {v:.2f}")
                                            else:
                                                st.write(f"- {k}: {v}")
                                
                                # Advanced Modeling Features
                                st.write("\n🔢 Advanced Modeling Features")
                                if 'modeling_features' in current_features:
                                    model_data = current_features['modeling_features']
                                    if isinstance(model_data, dict):
                                        st.write("Volatility & Lagged Features:")
                                        for k, v in model_data.items():
                                            if isinstance(v, (int, float)):
                                                st.write(f"- {k}: {v:.2f}")
                                            else:
                                                st.write(f"- {k}: {v}")
                                    
                                    # GARCH Volatility
                                    if 'garch_volatility' in market_data:
                                        if isinstance(market_data['garch_volatility'], (int, float)):
                                            st.write(f"\nGARCH Volatility: {market_data['garch_volatility']:.2f}")
                                        else:
                                            st.write(f"\nGARCH Volatility: {market_data['garch_volatility']}")
                                    
                                    # Lagged Features
                                    if 'lagged_features' in model_data:
                                        st.write("\nLagged Features:")
                                        for k, v in model_data['lagged_features'].items():
                                            if isinstance(v, (int, float)):
                                                st.write(f"- {k}: {v:.2f}")
                                            else:
                                                st.write(f"- {k}: {v}")
            
                                # Debug info for latest data date
                                if isinstance(current_features.get('price_data'), dict) and 'date' in current_features['price_data']:
                                    latest_date = datetime.strptime(current_features['price_data']['date'], '%Y-%m-%d').date()
                                elif isinstance(current_features.get('price_data'), list) and current_features['price_data']:
                                    latest_date = datetime.strptime(current_features['price_data'][-1]['date'], '%Y-%m-%d').date()
                                else:
                                    latest_date = None
                                if latest_date:
                                    st.info(f"Latest available price data is from: {latest_date}")
            
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
                        df = data_fetcher.fetch_stock_data(ticker, 
                                                         (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
                                                         datetime.now().strftime('%Y-%m-%d'))
                        
                        if not df.empty:
                            df = data_fetcher.calculate_technical_indicators(df)
                            fig = plot_technical_indicators(df)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Could not fetch technical data. Please check the ticker symbol and try again.")
            
            # Sentiment Analysis Tab
            with tabs[4]:
                st.header("Sentiment Analysis")
                if current_features.get('sentiment'):
                    st.subheader("Sentiment Timeline")
                    sentiment_data = [
                        {**entry, 'date': entry.get('timestamp')}
                        for entry in sentiment_analyzer.sentiment_history.get(ticker, [])
                    ]
                    timeline_fig = visualizer.create_sentiment_timeline(
                        sentiment_data,
                        ticker
                    )
                    st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    st.subheader("Sentiment Heatmap")
                    heatmap_fig = visualizer.create_sentiment_heatmap(
                        sentiment_data,
                        ticker
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    st.subheader("Current Sentiment Distribution")
                    radar_fig = visualizer.create_sentiment_radar(current_features.get('sentiment', {}), ticker)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    st.subheader("Trend Analysis")
                    trend_fig = visualizer.create_trend_analysis(
                        sentiment_data,
                        ticker
                    )
                    st.plotly_chart(trend_fig, use_container_width=True)
                else:
                    st.info("No sentiment data available for visualization.")
            
            # RAG Training Data Tab
            with tabs[5]:
                st.header("RAG Training Data Report")
                display_training_report(rag_records)
                plot_training_chart(rag_records)
                download_features_json(rag_records)
            
            # Evaluation section in sidebar
            st.sidebar.header("Evaluation")
            st.sidebar.markdown("After the week, enter the actual outcome to help the agent learn.")
            actual_action = st.sidebar.selectbox("Actual Action Outcome", ["Success", "Failure", "Unknown"])
            actual_return = st.sidebar.number_input("Actual Return (%)", value=0.0, step=0.01)
            if st.sidebar.button("Log Actual Outcome"):
                meta_agent.evaluate_and_log_result(
                    ticker,
                    timeframe,
                    {"actual_action": actual_action, "actual_return": actual_return}
                )
                st.sidebar.success("Outcome logged!")
        
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            traceback.print_exc()
            st.error(f"Error analyzing {ticker}: {e}")
            st.info("Please check if the ticker is valid and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit • Data from multiple financial, macro, and alternative sources</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 