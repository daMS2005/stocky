import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from src.ingestion.sentiment_analyzer import SentimentAnalyzer
from src.ingestion.fundamental_analyzer import FundamentalAnalyzer
from src.ingestion.data_fetcher import DataFetcher
from src.visualization.sentiment_visualizer import SentimentVisualizer
from src.gpt_agent.meta_agent import MetaAgentController
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize analyzers and fetchers
sentiment_analyzer = SentimentAnalyzer()
fundamental_analyzer = FundamentalAnalyzer()
data_fetcher = DataFetcher()
visualizer = SentimentVisualizer()
meta_agent = MetaAgentController()

# Set page config
st.set_page_config(
    page_title="Stock Financial AI Analyst",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Financial AI Analyst")
st.markdown("""
This tool analyzes a stock using technical, sentiment, fundamental, macro, and alternative data to provide a full investment report and trading recommendation. Enter a stock ticker below to get started.
""")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime(2022, 7, 1))
end_date = st.sidebar.date_input("End Date", datetime(2022, 7, 7))
timeframe = {"start": str(start_date), "end": str(end_date)}

# Main content
if ticker:
    try:
        with st.spinner(f"Analyzing {ticker} for {start_date} to {end_date}..."):
            # Fetch all features
            features = data_fetcher.get_all_features(
                ticker,
                sentiment_analyzer=sentiment_analyzer,
                fundamental_analyzer=fundamental_analyzer
            )
            
            # Check for errors
            if 'error' in features:
                st.error(features['error'])
                st.info("Please check if the ticker is valid and try again.")
                st.stop()
            
            # Check if we have enough data
            if not features.get('price_data'):
                st.error(f"No price data available for {ticker}")
                st.info("Please try a different ticker or date range.")
                st.stop()
            
            # Run meta-agent
            result = meta_agent.make_prediction(ticker, timeframe, features)
            structured = result["structured"]
            report = result["report"]
            feedback = result["feedback"]

            # Tabs: Report, JSON, Feedback, Visualizations
            tab1, tab2, tab3, tab4 = st.tabs([
                "Investment Report", "Structured JSON", "Feedback Context", "Sentiment Visualizations"
            ])

            with tab1:
                st.subheader("Natural Language Investment Report")
                st.markdown(report)

            with tab2:
                st.subheader("Structured JSON Output")
                # Format the recommendation for display
                if structured.get('prediction'):
                    prediction = structured['prediction']
                    # Format numeric fields with proper handling of None values
                    for field in ['entry_price', 'target_price', 'stop_loss']:
                        if prediction.get(field) is not None:
                            prediction[field] = f"${prediction[field]:.2f}"
                        else:
                            prediction[field] = "N/A"
                st.json(structured)

            with tab3:
                st.subheader("Feedback from Similar Past Cases")
                st.code(feedback)

            with tab4:
                if features.get('sentiment'):
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
                    radar_fig = visualizer.create_sentiment_radar(features.get('sentiment', {}), ticker)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    st.subheader("Trend Analysis")
                    trend_fig = visualizer.create_trend_analysis(
                        sentiment_data,
                        ticker
                    )
                    st.plotly_chart(trend_fig, use_container_width=True)
                else:
                    st.info("No sentiment data available for visualization.")

            # Evaluation section
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
        st.error(f"Error analyzing {ticker}: {str(e)}")
        st.info("Please check if the ticker is valid and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • Data from multiple financial, macro, and alternative sources</p>
</div>
""", unsafe_allow_html=True) 