"""Embedding model interfaces."""
import hashlib
import os
from typing import List, Protocol
import numpy as np
import requests


class EmbeddingModel(Protocol):
    """Embedding model interface."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts into vectors."""
        ...
    
    @property
    def dim(self) -> int:
        """Return embedding dimension."""
        ...


class MockEmbeddingModel:
    """Mock embedding model for testing.
    
    Note: Mock vectors have no semantic information.
    Retrieval quality tests cannot verify real effectiveness.
    """
    
    def __init__(self, dim: int = 1536):
        self._dim = dim
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings using hash."""
        embeddings = []
        for text in texts:
            # Generate diverse vectors by hashing text with multiple seeds
            vec = np.zeros(self._dim, dtype=np.float32)
            for i in range(self._dim):
                hash_input = f"{text}:{i}"
                hash_bytes = hashlib.sha256(hash_input.encode()).digest()
                # Convert first 4 bytes to float in [-1, 1]
                val = int.from_bytes(hash_bytes[:4], 'big') / (2**32 - 1)
                vec[i] = val * 2 - 1
            
            # Normalize to unit length (for cosine similarity)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            embeddings.append(vec.tolist())
        
        return embeddings
    
    @property
    def dim(self) -> int:
        return self._dim


class OllamaEmbeddingModel:
    """Ollama embedding model (local)."""
    
    def __init__(self, model: str = "qwen3-embedding:0.6b", base_url: str = "http://localhost:11434"):
        """Initialize Ollama embedding model.
        
        Args:
            model: Model name (default: qwen3-embedding:0.6b)
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self._dim = None  # Will be determined on first call
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Ollama API."""
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        
        embeddings = response.json()["embeddings"]
        
        # Set dimension on first call
        if self._dim is None:
            self._dim = len(embeddings[0])
        
        return embeddings
    
    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe dimension with a test call
            test_embedding = self.embed(["test"])[0]
            self._dim = len(test_embedding)
        return self._dim


class OpenAIEmbeddingModel:
    """OpenAI embedding model (text-embedding-ada-002)."""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002"):
        """Initialize OpenAI embedding model.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model name (default: text-embedding-ada-002)
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.model = model
        self._dim = 1536  # ada-002 dimension
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI API."""
        # OpenAI supports batch embedding
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        return [item.embedding for item in response.data]
    
    @property
    def dim(self) -> int:
        return self._dim


class LocalBGEEmbeddingModel:
    """Local BGE embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5"):
        """Initialize local BGE model.
        
        Args:
            model_name: HuggingFace model name (default: BAAI/bge-base-zh-v1.5)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using local BGE model."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    @property
    def dim(self) -> int:
        return self._dim


def create_embedding_model(model_type: str = "mock", **kwargs) -> EmbeddingModel:
    """Factory function to create embedding model.
    
    Args:
        model_type: "mock", "ollama", "openai", or "local_bge"
        **kwargs: Model-specific arguments
    
    Returns:
        EmbeddingModel instance
    """
    if model_type == "mock":
        return MockEmbeddingModel(**kwargs)
    elif model_type == "ollama":
        return OllamaEmbeddingModel(**kwargs)
    elif model_type == "openai":
        return OpenAIEmbeddingModel(**kwargs)
    elif model_type == "local_bge":
        return LocalBGEEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
