import os
import pickle
from typing import List, Dict, Any
import numpy as np
from datetime import date

# --- Base Vector Store Interface ---
class BaseVectorStore:
    def add_record(self, record: dict):
        raise NotImplementedError

    def query_similar(self, query_embedding: list, top_n: int = 5) -> list:
        raise NotImplementedError

    def get_all(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

# --- Local (Pickle) Vector Store ---
class LocalVectorStore(BaseVectorStore):
    def __init__(self, db_path: str = 'rag_db.pkl'):
        self.db_path = db_path
        self.records: List[Dict[str, Any]] = self._load_db()

    def _load_db(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                return pickle.load(f)
        return []

    def _save_db(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.records, f)

    def add_record(self, record: dict):
        self.records.append(record)
        self._save_db()

    def get_all(self) -> List[Dict[str, Any]]:
        return self.records

    def query_similar(self, query_embedding: list, top_n: int = 5) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        sims = []
        q = np.array(query_embedding)
        for rec in self.records:
            # Aggregate all embeddings for the week (mean)
            if rec.get('embeddings'):
                emb = np.mean(np.array(rec['embeddings']), axis=0)
                sim = np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-8)
                sims.append((sim, rec))
        sims.sort(reverse=True, key=lambda x: x[0])
        return [rec for _, rec in sims[:top_n]]

# --- FAISS Vector Store ---
class FaissVectorStore(BaseVectorStore):
    def __init__(self, db_path: str = 'faiss_index', meta_path: str = 'faiss_meta.pkl', embedding_dim: int = 1536):
        import faiss
        import openai
        self.db_path = db_path
        self.meta_path = meta_path
        self.embedding_dim = embedding_dim
        self.faiss = faiss
        self.openai = openai
        # Load or create FAISS index
        if os.path.exists(self.db_path):
            self.index = faiss.read_index(self.db_path)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        # Load or create metadata
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []

    def _save(self):
        self.faiss.write_index(self.index, self.db_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def add_record(self, record: dict):
        # Aggregate all embeddings for the week (mean)
        if record.get('embeddings'):
            emb = np.mean(np.array(record['embeddings']), axis=0).astype('float32')
            self.index.add(np.expand_dims(emb, axis=0))
            self.metadata.append(record)
            self._save()

    def update_record(self, old_record: dict, new_record: dict):
        """
        Update an existing record in the FAISS index and metadata.
        
        Args:
            old_record: The existing record to update
            new_record: The new record with updated data
        """
        # Find the index of the old record
        try:
            idx = self.metadata.index(old_record)
            
            # Remove old record from index
            if old_record.get('embeddings'):
                old_emb = np.mean(np.array(old_record['embeddings']), axis=0).astype('float32')
                # Create a new index without the old record
                new_index = self.faiss.IndexFlatL2(self.embedding_dim)
                for i, rec in enumerate(self.metadata):
                    if i != idx:
                        emb = np.mean(np.array(rec['embeddings']), axis=0).astype('float32')
                        new_index.add(np.expand_dims(emb, axis=0))
                self.index = new_index
            
            # Update metadata
            self.metadata[idx] = new_record
            
            # Add new record to index
            if new_record.get('embeddings'):
                new_emb = np.mean(np.array(new_record['embeddings']), axis=0).astype('float32')
                self.index.add(np.expand_dims(new_emb, axis=0))
            
            # Save changes
            self._save()
            
        except ValueError:
            print(f"Warning: Could not find record to update in metadata")
            # If record not found, add as new
            self.add_record(new_record)

    def get_all(self) -> List[Dict[str, Any]]:
        return self.metadata

    def query_similar(self, query_embedding, k=3):
        """
        Query the vector store for similar records using FAISS.
        
        Args:
            query_embedding: The embedding vector to search for (list of floats)
            k: Number of similar records to return
        
        Returns:
            List of similar records
        """
        if not self.index:
            return []
        
        # Convert query embedding to numpy array if it's not already
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype='float32')
        
        # Reshape for FAISS if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search using FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the corresponding records
        similar_records = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                similar_records.append(self.metadata[idx])
        
        return similar_records

# --- OpenAI Vector Store Stub ---
class OpenAIVectorStore(BaseVectorStore):
    def __init__(self, *args, **kwargs):
        # Placeholder for OpenAI vector store initialization
        pass

    def add_record(self, record: dict):
        # TODO: Implement OpenAI vector store add
        raise NotImplementedError("OpenAI Vector Store integration coming soon.")

    def get_all(self) -> List[Dict[str, Any]]:
        # TODO: Implement OpenAI vector store retrieval
        raise NotImplementedError("OpenAI Vector Store integration coming soon.")

    def query_similar(self, query_embedding: list, top_n: int = 5) -> List[Dict[str, Any]]:
        # TODO: Implement OpenAI vector store similarity search
        raise NotImplementedError("OpenAI Vector Store integration coming soon.")

# --- Factory Function ---
def get_vector_store(store_type='local', **kwargs) -> BaseVectorStore:
    if store_type == 'openai':
        return OpenAIVectorStore(**kwargs)
    elif store_type == 'faiss':
        return FaissVectorStore(**kwargs)
    else:
        return LocalVectorStore(**kwargs)

# --- Usage Example ---
# vector_store = get_vector_store('faiss')
# vector_store.add_record({...})
# results = vector_store.query_similar(query_text, top_n=5)

# --- RAG Database ---
class RAGDatabase:
    def __init__(self, db_path: str = 'rag_db.pkl'):
        self.db_path = db_path
        self.records: List[Dict[str, Any]] = self._load_db()

    def _load_db(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                return pickle.load(f)
        return []

    def _save_db(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.records, f)

    def add_week(self, week_start: date, ticker: str, features: Dict, texts: List[str], 
                embeddings: List[List[float]], price_data: Dict[str, float]):
        """
        Add a week's data to the RAG database.
        
        Args:
            week_start: Start date of the week
            ticker: Stock ticker symbol
            features: Technical and fundamental features
            texts: News headlines and other text data
            embeddings: Text embeddings
            price_data: Dictionary containing:
                - open_price: Price at start of week
                - close_price: Price at end of week
                - weekly_return: Percentage return for the week
        """
        record = {
            'week_start': week_start,
            'ticker': ticker,
            'features': features,
            'texts': texts,
            'embeddings': embeddings,
            'price_data': price_data
        }
        self.records.append(record)
        self._save_db()

    def get_all(self) -> List[Dict[str, Any]]:
        return self.records

    # --- Retriever ---
    def retrieve_similar(self, query_embedding: List[float], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find the top-N most similar weeks to the query embedding (cosine similarity).
        Returns a list of records.
        """
        if not self.records:
            return []
        sims = []
        q = np.array(query_embedding)
        for rec in self.records:
            # Aggregate all embeddings for the week (mean)
            if rec['embeddings']:
                emb = np.mean(np.array(rec['embeddings']), axis=0)
                sim = np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-8)
                sims.append((sim, rec))
        sims.sort(reverse=True, key=lambda x: x[0])
        return [rec for _, rec in sims[:top_n]]

# --- Note: For OpenAI-native RAG ---
# In the future, you can use OpenAI's vector store API (when available) to store and query embeddings natively.
# For now, this local implementation is compatible with OpenAI embeddings and can be swapped out later. 