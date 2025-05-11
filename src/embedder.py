import os
import pickle
from typing import Dict

# Placeholder for your actual embedding function (e.g., OpenAI, HuggingFace, etc.)
def embed_text(text: str) -> list:
    # Replace this with your real embedding logic
    # For example: return openai_embedder.embed(text)
    return [float(ord(c)) for c in text[:10]]  # Dummy: vector of first 10 chars

class EmbeddingCache:
    def __init__(self, cache_path: str = 'embedding_cache.pkl'):
        self.cache_path = cache_path
        self.cache: Dict[str, list] = self._load_cache()

    def _load_cache(self) -> Dict[str, list]:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def get_embedding(self, text: str) -> list:
        if text in self.cache:
            return self.cache[text]
        vec = embed_text(text)
        self.cache[text] = vec
        self._save_cache()
        return vec

    def update_cache(self, texts):
        """Embed and cache a list of new texts."""
        for text in texts:
            self.get_embedding(text)
        self._save_cache()

# Example usage:
if __name__ == '__main__':
    cache = EmbeddingCache()
    sample_texts = ["Apple earnings beat expectations", "Tesla launches new model"]
    for t in sample_texts:
        vec = cache.get_embedding(t)
        print(f"Embedding for '{t}': {vec}") 