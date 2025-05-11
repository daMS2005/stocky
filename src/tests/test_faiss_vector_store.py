from src.rag_db import get_vector_store
import numpy as np
from datetime import date

# Dummy data for testing
week_start = date(2022, 6, 20)
ticker = 'AAPL'
features = {'rsi': 46.75, 'pe_ratio': 28.5}
texts = [
    "Apple releases new iPhone model.",
    "AAPL Q2 earnings beat expectations."
]
# Use random embeddings for testing add_record (replace with real ones in production)
embeddings = [np.random.rand(1536).tolist() for _ in texts]
outcome = {'return': 0.012, 'action': 'Buy'}

if __name__ == '__main__':
    # Initialize FAISS vector store
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    # Add a record
    print('Adding record...')
    vector_store.add_record({
        'week_start': week_start,
        'ticker': ticker,
        'features': features,
        'texts': texts,
        'embeddings': embeddings,
        'outcome': outcome
    })
    print('Record added.')

    # Query similar records using OpenAI embedding
    query = "Apple launches new product"
    print(f'Querying similar records for: "{query}"')
    results = vector_store.query_similar(query, top_n=3)
    print('Top similar records:')
    for rec in results:
        print(f"Week: {rec['week_start']}, Ticker: {rec['ticker']}, Features: {rec['features']}, Outcome: {rec['outcome']}") 