from datetime import date, timedelta, datetime as dt
from src.ingestion.data_fetcher import DataFetcher
from src.embedder import EmbeddingCache
from src.rag_db import get_vector_store
import openai
import numpy as np
import argparse
import os
import json
import pandas as pd

# --- CLI ---
def parse_args():
    parser = argparse.ArgumentParser(description='Populate RAG DB for a given ticker and date range.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker (default: AAPL)')
    parser.add_argument('--start', type=str, default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2022-06-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--force-update', action='store_true', help='Force update all records even if they exist')
    return parser.parse_args()

# --- Helper ---
def week_key(ticker, week_start):
    return f"{ticker}_{week_start}"

def get_existing_record(vector_store, ticker, week_start):
    """Get existing record if it exists."""
    key = week_key(ticker, week_start)
    for rec in vector_store.get_all():
        if week_key(rec['ticker'], rec['week_start']) == key:
            return rec
    return None

def features_changed(old_features, new_features):
    """Compare old and new features to determine if an update is needed."""
    def normalize_value(v):
        if isinstance(v, (np.ndarray, list)):
            return json.dumps(v)
        return v

    def compare_dicts(d1, d2):
        if set(d1.keys()) != set(d2.keys()):
            return True
        
        for k in d1:
            v1 = normalize_value(d1[k])
            v2 = normalize_value(d2[k])
            
            if isinstance(v1, dict) and isinstance(v2, dict):
                if compare_dicts(v1, v2):
                    return True
            elif v1 != v2:
                return True
        return False

    return compare_dicts(old_features, new_features)

def populate_rag_db(ticker, start, end, force_update=False):
    TICKER = ticker.upper()
    START_DATE = dt.strptime(start, '%Y-%m-%d').date()
    END_DATE = dt.strptime(end, '%Y-%m-%d').date()
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536

    # Initialize components
    vector_store = get_vector_store('faiss', embedding_dim=EMBEDDING_DIM)
    embedder = EmbeddingCache()
    data_fetcher = DataFetcher()
    
    current = START_DATE
    while current <= END_DATE:
        week_start = current
        week_end = week_start + timedelta(days=4)
        print(f"\nProcessing week {week_key(TICKER, week_start)}...")
        # 1. Get price data and features
        lookback_start = week_start - timedelta(days=60)
        df = data_fetcher.fetch_stock_data(TICKER, lookback_start.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d'))
        if df.empty:
            print(f"No data available for week of {week_start}, skipping.")
            current += timedelta(days=7)
            continue
        df = data_fetcher.calculate_technical_indicators(df)
        week_df = df[df.index.date >= week_start]
        if week_df.empty:
            print(f"No data available for week of {week_start}, skipping.")
            current += timedelta(days=7)
            continue
        new_features = {
            'technical': {
                'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else None,
                'macd': float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else None,
                'macd_signal': float(df['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]) else None,
                'bollinger_upper': float(df['BB_Upper'].iloc[-1]) if 'BB_Upper' in df.columns and not pd.isna(df['BB_Upper'].iloc[-1]) else None,
                'bollinger_lower': float(df['BB_Lower'].iloc[-1]) if 'BB_Lower' in df.columns and not pd.isna(df['BB_Lower'].iloc[-1]) else None,
                'bollinger_middle': float(df['BB_Middle'].iloc[-1]) if 'BB_Middle' in df.columns and not pd.isna(df['BB_Middle'].iloc[-1]) else None,
                'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns and not pd.isna(df['Volume'].iloc[-1]) else None,
                'volume_ma': float(df['Volume_MA'].iloc[-1]) if 'Volume_MA' in df.columns and not pd.isna(df['Volume_MA'].iloc[-1]) else None,
                'volume_ratio': float(df['Volume_Ratio'].iloc[-1]) if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]) else None,
                'garch_volatility': data_fetcher.get_garch_volatility(df)
            },
            'price_data': {
                'open_price': float(week_df['Open'].iloc[0]),
                'close_price': float(week_df['Close'].iloc[-1]),
                'high_price': float(week_df['High'].max()),
                'low_price': float(week_df['Low'].min()),
                'weekly_return': float(((week_df['Close'].iloc[-1] - week_df['Open'].iloc[0]) / week_df['Open'].iloc[0]) * 100),
                'daily_returns': week_df['Close'].pct_change().dropna().tolist(),
                'volume': float(week_df['Volume'].sum())
            }
        }
        weekly_return = new_features['price_data']['weekly_return']
        if weekly_return > 2.0:
            outcome = "STRONG_BUY"
        elif weekly_return > 0.5:
            outcome = "BUY"
        elif weekly_return < -2.0:
            outcome = "STRONG_SELL"
        elif weekly_return < -0.5:
            outcome = "SELL"
        else:
            outcome = "HOLD"
        try:
            ratios = data_fetcher.get_financial_ratios(TICKER)
            new_features['fundamental'] = {
                'debt_to_equity': ratios.get('debt_to_equity'),
                'return_on_equity': ratios.get('return_on_equity'),
                'return_on_assets': ratios.get('return_on_assets'),
                'beta': ratios.get('beta'),
                'dividend_yield': ratios.get('dividend_yield'),
                'institutional_ownership': ratios.get('institutional_ownership'),
                'short_interest_ratio': ratios.get('short_interest_ratio')
            }
        except Exception as e:
            print(f"Warning: Could not fetch fundamental data: {str(e)}")
            new_features['fundamental'] = {}
        try:
            new_features['macro'] = {
                'country_risk': data_fetcher.get_country_risk(TICKER),
                'fx_rates': data_fetcher.get_fx_rates(),
                'vix': data_fetcher.get_vix_from_fred()
            }
        except Exception as e:
            print(f"Warning: Could not fetch macro data: {str(e)}")
            new_features['macro'] = {}
        texts = [
            f"{TICKER} news headline for week of {week_start}",
            f"{TICKER} earnings update for week of {week_start}"
        ]
        embeddings = []
        for text in texts:
            response = openai.embeddings.create(model=EMBEDDING_MODEL, input=text)
            emb = response.data[0].embedding
            embeddings.append(np.array(emb, dtype='float32'))
        record = {
            'week_start': week_start,
            'ticker': TICKER,
            'features': new_features,
            'texts': texts,
            'embeddings': [e.tolist() for e in embeddings],
            'price_data': new_features['price_data'],
            'outcome': outcome
        }
        existing_record = get_existing_record(vector_store, TICKER, week_start)
        if existing_record and not force_update:
            if not features_changed(existing_record['features'], new_features):
                print(f"Week {week_key(TICKER, week_start)} already in DB with same features, skipping.")
                current += timedelta(days=7)
                continue
            else:
                print(f"Updating existing record for week {week_key(TICKER, week_start)}...")
        if existing_record:
            vector_store.update_record(existing_record, record)
            print(f"Updated week {week_key(TICKER, week_start)} in DB.")
        else:
            vector_store.add_record(record)
            print(f"Added new week {week_key(TICKER, week_start)} to DB.")
        print(f"Price Movement: Open ${new_features['price_data']['open_price']:.2f} → Close ${new_features['price_data']['close_price']:.2f} ({new_features['price_data']['weekly_return']:+.2f}%)")
        print(f"Technical Indicators: RSI={new_features['technical']['rsi']:.2f}, MACD={new_features['technical']['macd']:.2f}")
        if new_features.get('macro', {}).get('country_risk') is not None:
            print(f"Country Risk: {new_features['macro']['country_risk']}, VIX: {new_features['macro'].get('vix')}")
        current += timedelta(days=7)
    print("\nDone populating RAG DB.")

# --- CLI ---
if __name__ == '__main__':
    args = parse_args()
    populate_rag_db(args.ticker, args.start, args.end, args.force_update) 