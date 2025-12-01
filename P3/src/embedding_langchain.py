from typing import List
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class MiniLMEmbeddings(Embeddings):
    """LangChain-compatible embedding wrapper for all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return self.model.encode(texts, convert_to_numpy=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.model.encode(text, convert_to_numpy=False).tolist()
