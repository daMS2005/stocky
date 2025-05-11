import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_db import get_vector_store

def check_latest_rag_db(ticker):
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    all_records = [r for r in vector_store.get_all() if r['ticker'] == ticker]
    if not all_records:
        print(f"No RAG data found for {ticker}.")
        return
    latest_record = max(all_records, key=lambda x: x['week_start'])
    print(f"Latest RAG DB record for {ticker}:")
    print(f"Week Start: {latest_record['week_start']}")
    print(f"Features: {latest_record['features']}")
    print(f"Price Data: {latest_record['price_data']}")

if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper()
    check_latest_rag_db(ticker) 