from sentence_transformers import SentenceTransformer
import numpy as np

from config import EMBEDDING_MODEL

class Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL):
        """Initialize embedder with the specified model."""
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text):
        """Embed a single text into a vector."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        embedding = self.model.encode(text)
        return embedding.astype(np.float32)
    
    def embed_documents(self, documents):
        """Embed a list of documents into vectors."""
        if not documents:
            return []
        
        # Extract text from documents if they are dict-like
        texts = []
        for doc in documents:
            if isinstance(doc, dict) and 'content' in doc:
                texts.append(doc['content'])
            elif isinstance(doc, str):
                texts.append(doc)
            else:
                raise ValueError("Documents must be strings or dicts with 'content' key")
        
        embeddings = self.model.encode(texts)
        return [e.astype(np.float32) for e in embeddings] 