"""Reranker interfaces for retrieval results."""
from typing import List, Protocol
import requests


class Reranker(Protocol):
    """Reranker interface."""
    
    def rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        """Rerank chunks by query.
        
        Args:
            query: Query text
            chunks: List of chunks with 'content' field
        
        Returns:
            Reranked chunks with 'score' field
        """
        ...


class MockReranker:
    """Mock reranker that preserves original order."""
    
    def rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        """Return chunks in original order."""
        return chunks


class OllamaReranker:
    """Ollama reranker using local model (if available).
    
    Note: Ollama doesn't have a dedicated reranker API yet.
    This uses a workaround with embedding similarity.
    """
    
    def __init__(self, model: str = "qwen3-embedding:0.6b", base_url: str = "http://localhost:11434"):
        """Initialize Ollama reranker.
        
        Args:
            model: Embedding model name
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
    
    def rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        """Rerank chunks by embedding similarity to query."""
        if not chunks:
            return chunks
        
        # Get query embedding
        query_response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": [query]}
        )
        query_response.raise_for_status()
        query_embedding = query_response.json()["embeddings"][0]
        
        # Get chunk embeddings
        chunk_texts = [c["content"] for c in chunks]
        chunks_response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": chunk_texts}
        )
        chunks_response.raise_for_status()
        chunk_embeddings = chunks_response.json()["embeddings"]
        
        # Calculate cosine similarity
        import numpy as np
        query_vec = np.array(query_embedding)
        
        reranked = []
        for chunk, emb in zip(chunks, chunk_embeddings):
            chunk_vec = np.array(emb)
            similarity = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
            reranked.append({**chunk, "score": float(similarity)})
        
        # Sort by similarity descending
        reranked.sort(key=lambda x: x["score"], reverse=True)
        
        return reranked


def create_reranker(reranker_type: str = "mock", **kwargs) -> Reranker:
    """Factory function to create reranker.
    
    Args:
        reranker_type: "mock" or "ollama"
        **kwargs: Reranker-specific arguments
    
    Returns:
        Reranker instance
    """
    if reranker_type == "mock":
        return MockReranker()
    elif reranker_type == "ollama":
        return OllamaReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
