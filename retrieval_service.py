#!/usr/bin/env python3
"""Retrieval service with reranking support."""
import json
import urllib.request
from typing import List, Dict, Any, Optional
import psycopg2
from datetime import datetime

from storage import ChunkRepository, RetrievedChunk


class OllamaEmbedding:
    """Ollama embedding client without numpy dependency."""
    
    def __init__(self, model: str = "qwen3-embedding:4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Ollama API."""
        data = json.dumps({"model": self.model, "input": texts}).encode('utf-8')
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
        embeddings = result["embeddings"]
        # Truncate to 2000 dims (pgvector index limit)
        return [e[:2000] for e in embeddings]


class OllamaReranker:
    """Ollama reranker using embedding similarity."""
    
    def __init__(self, embedding_client: OllamaEmbedding):
        self.embedding_client = embedding_client
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank chunks by embedding similarity to query."""
        if not chunks:
            return chunks
        
        # Get query embedding
        query_embedding = self.embedding_client.embed([query])[0]
        
        # Get chunk embeddings
        chunk_texts = [c["content"] for c in chunks]
        chunk_embeddings = self.embedding_client.embed(chunk_texts)
        
        # Calculate cosine similarity (without numpy)
        def dot_product(a, b):
            return sum(x * y for x, y in zip(a, b))
        
        def norm(v):
            return sum(x * x for x in v) ** 0.5
        
        def cosine_similarity(a, b):
            return dot_product(a, b) / (norm(a) * norm(b))
        
        # Score each chunk
        for chunk, chunk_emb in zip(chunks, chunk_embeddings):
            chunk["score"] = cosine_similarity(query_embedding, chunk_emb)
        
        # Sort by score descending
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks


class RetrievalService:
    """High-level retrieval service with reranking."""
    
    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        embedding_model: str = "qwen3-embedding:4b",
        reranker_type: str = "ollama"
    ):
        """Initialize retrieval service.
        
        Args:
            conn: Database connection
            embedding_model: Embedding model name
            reranker_type: Reranker type ("mock" or "ollama")
        """
        self.repo = ChunkRepository(conn)
        self.embedding_client = OllamaEmbedding(model=embedding_model)
        
        if reranker_type == "ollama":
            self.reranker = OllamaReranker(self.embedding_client)
        else:
            self.reranker = None
    
    def search(
        self,
        query: str,
        limit: int = 10,
        rerank: bool = True,
        available_before: Optional[datetime] = None,
        fts_weight: float = 0.7,
        vector_weight: float = 0.3,
        rerank_top_k: int = 20
    ) -> List[RetrievedChunk]:
        """Search with optional reranking.
        
        Args:
            query: Search query
            limit: Final number of results
            rerank: Whether to apply reranking
            available_before: PIT cutoff
            fts_weight: FTS weight for hybrid search
            vector_weight: Vector weight for hybrid search
            rerank_top_k: Number of candidates to rerank
        
        Returns:
            List of retrieved chunks
        """
        # Get embedding for query
        query_embedding = self.embedding_client.embed([query])[0]
        
        # Initial hybrid search
        candidates = self.repo.hybrid_search(
            query,
            embedding=query_embedding,
            limit=rerank_top_k if rerank else limit,
            available_before=available_before,
            fts_weight=fts_weight,
            vector_weight=vector_weight
        )
        
        if not rerank or not candidates:
            return candidates[:limit]
        
        # Convert to dict for reranking
        chunks_dict = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content,
                "section_path": c.section_path,
                "chunk_index": c.chunk_index,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
                "token_count": c.token_count,
                "available_at": c.available_at,
                "score": c.score
            }
            for c in candidates
        ]
        
        # Rerank
        reranked = self.reranker.rerank(query, chunks_dict)
        
        # Convert back to RetrievedChunk
        results = []
        for item in reranked[:limit]:
            results.append(RetrievedChunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                content=item["content"],
                section_path=item["section_path"],
                chunk_index=item["chunk_index"],
                start_offset=item["start_offset"],
                end_offset=item["end_offset"],
                token_count=item["token_count"],
                available_at=item["available_at"],
                score=item["score"]
            ))
        
        return results


def create_retrieval_service(
    dsn: str = "postgresql://travis@localhost/nooshare_dev",
    **kwargs
) -> RetrievalService:
    """Factory function to create retrieval service.
    
    Args:
        dsn: Database connection string
        **kwargs: Additional arguments for RetrievalService
    
    Returns:
        RetrievalService instance
    """
    conn = psycopg2.connect(dsn)
    return RetrievalService(conn, **kwargs)
