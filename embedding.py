"""Embedding interface for text vectorization."""
from typing import List, Protocol
import hashlib


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""
        ...
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        ...


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, dim: int = 1536):
        self._dimension = dim
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Use hash to generate deterministic but unique vectors
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Convert bytes to floats in [-1, 1]
            vector = []
            for i in range(0, min(len(hash_bytes) * 4, self._dimension), 4):
                if i + 4 <= len(hash_bytes):
                    # Use 4 bytes to generate a float
                    val = int.from_bytes(hash_bytes[i:i+4], 'big')
                    normalized = (val / (2**32 - 1)) * 2 - 1
                    vector.append(normalized)
                else:
                    vector.append(0.0)
            # Pad with zeros if needed
            while len(vector) < self._dimension:
                vector.append(0.0)
            embeddings.append(vector)
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension
