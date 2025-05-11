from datetime import date
from src.feature_extractor import extract_numerical_features
from src.embedder import EmbeddingCache
from src.sequence_builder import build_weekly_feature_vector

if __name__ == '__main__':
    ticker = 'AAPL'
    week_start = date(2022, 6, 20)
    week_end = date(2022, 6, 24)

    # 1. Extract numerical features
    num_features = extract_numerical_features(ticker, week_start, week_end)
    print('Numerical features:', num_features)

    # 2. Embed a list of texts for the week
    texts = [
        "Apple releases new iPhone model.",
        "AAPL Q2 earnings beat expectations.",
        "Federal Reserve raises interest rates."
    ]
    embedder = EmbeddingCache()
    text_embeddings = [embedder.get_embedding(t) for t in texts]
    print('Text embeddings:', text_embeddings)

    # 3. Build the combined weekly feature vector
    combined = build_weekly_feature_vector(num_features, text_embeddings)
    print('Combined weekly feature vector:')
    for k, v in combined.items():
        print(f"  {k}: {v}") 