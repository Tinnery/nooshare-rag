"""Embedding interface for text vectorization."""
from typing import List, Protocol
import hashlib
import math


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
            # Use multiple hashes to generate enough bytes for the full vector
            vector = []
            hash_idx = 0
            
            while len(vector) < self._dimension:
                # Generate a new hash for each chunk
                hash_input = f"{text}:{hash_idx}".encode()
                hash_bytes = hashlib.sha256(hash_input).digest()
                
                # Convert 32 bytes to 8 floats (4 bytes each)
                for i in range(0, 32, 4):
                    if len(vector) >= self._dimension:
                        break
                    val = int.from_bytes(hash_bytes[i:i+4], 'big')
                    # Normalize to [-1, 1]
                    normalized = (val / (2**32 - 1)) * 2 - 1
                    vector.append(normalized)
                
                hash_idx += 1
            
            # Normalize the vector to unit length (cosine similarity)
            magnitude = math.sqrt(sum(v * v for v in vector))
            if magnitude > 0:
                vector = [v / magnitude for v in vector]
            
            embeddings.append(vector)
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension
