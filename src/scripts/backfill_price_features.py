from src.rag_db import get_vector_store
import yfinance as yf
from datetime import timedelta
import pickle


def get_weekly_open_close(ticker, week_start, week_end):
    df = yf.download(ticker, start=week_start, end=week_end + timedelta(days=1))
    if df.empty:
        return None, None, None, None
    open_price = float(df.iloc[0]['Open'])
    close_price = float(df.iloc[-1]['Close'])
    high_price = float(df['High'].max())
    low_price = float(df['Low'].min())
    return open_price, close_price, high_price, low_price

if __name__ == '__main__':
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    updated = 0
    for rec in vector_store.metadata:
        features = rec['features']
        # Only update if missing
        if not all(k in features for k in ['open_price', 'close_price', 'high_price', 'low_price']):
            week_start = rec['week_start']
            week_end = week_start + timedelta(days=4)
            open_p, close_p, high_p, low_p = get_weekly_open_close(rec['ticker'], week_start, week_end)
            features['open_price'] = open_p
            features['close_price'] = close_p
            features['high_price'] = high_p
            features['low_price'] = low_p
            updated += 1
    if updated:
        with open('faiss_meta.pkl', 'wb') as f:
            pickle.dump(vector_store.metadata, f)
    print(f"Updated {updated} records with weekly price features.") 