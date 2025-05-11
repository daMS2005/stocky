from typing import Dict, List, Any

# This function merges numerical features and text embeddings into a single vector or dict

def build_weekly_feature_vector(numerical_features: Dict[str, float], text_embeddings: List[List[float]]) -> Dict[str, Any]:
    """
    Combine all actual numerical features and a list of text embeddings into a single feature vector for the week.
    Returns a dict with all features, including the full list of text embeddings.
    """
    combined = dict(numerical_features)
    combined['text_embeddings'] = text_embeddings
    combined['text_embedding_count'] = len(text_embeddings)
    # Optionally, add mean embedding for LLM context
    if text_embeddings:
        embedding_dim = len(text_embeddings[0])
        mean_embedding = [sum(vec[i] for vec in text_embeddings) / len(text_embeddings) for i in range(embedding_dim)]
    else:
        mean_embedding = []
    combined['text_embedding_mean'] = mean_embedding
    return combined 