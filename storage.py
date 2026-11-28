"""PostgreSQL storage layer for chunks and embeddings."""
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass

from document import Document, DocumentMetadata, DocumentType
from chunker import Chunk


@dataclass
class RetrievedChunk:
    """A chunk retrieved from storage with score."""
    chunk_id: str
    doc_id: str
    content: str
    section_path: str
    chunk_index: int
    start_offset: int
    end_offset: int
    token_count: int
    score: float  # Similarity or rank score
    available_at: datetime


class ChunkRepository:
    """Repository for chunk storage and retrieval."""
    
    def __init__(self, conn: psycopg2.extensions.connection):
        self.conn = conn
    
    def save_document(self, doc: Document) -> None:
        """Save a document and its chunks."""
        with self.conn.cursor() as cur:
            # Insert document metadata
            cur.execute("""
                INSERT INTO documents (
                    doc_id, doc_type, symbol, fiscal_year, fiscal_period,
                    published_at, available_at, source_path, checksum
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    updated_at = NOW()
            """, (
                doc.metadata.doc_id,
                doc.metadata.doc_type.value,
                doc.metadata.symbol,
                doc.metadata.fiscal_year,
                doc.metadata.fiscal_period,
                doc.metadata.published_at,
                doc.metadata.available_at,
                doc.metadata.source_path,
                doc.metadata.checksum
            ))
        
        self.conn.commit()
    
    def save_chunks(self, chunks: List[Chunk]) -> None:
        """Save chunks in batch."""
        if not chunks:
            return
        
        with self.conn.cursor() as cur:
            values = [
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.content,
                    chunk.section_path,
                    chunk.chunk_index,
                    chunk.start_offset,
                    chunk.end_offset,
                    chunk.token_count
                )
                for chunk in chunks
            ]
            
            execute_values(
                cur,
                """
                    INSERT INTO chunks (
                        chunk_id, doc_id, content, section_path,
                        chunk_index, start_offset, end_offset, token_count
                    ) VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        updated_at = NOW()
                """,
                values
            )
        
        self.conn.commit()
    
    def save_embeddings(self, chunk_ids: List[str], embeddings: List[List[float]]) -> None:
        """Save embeddings for chunks."""
        if not chunk_ids or not embeddings:
            return
        
        with self.conn.cursor() as cur:
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                cur.execute("""
                    UPDATE chunks
                    SET embedding = %s::vector, updated_at = NOW()
                    WHERE chunk_id = %s
                """, (embedding, chunk_id))
        
        self.conn.commit()
    
    def full_text_search(
        self,
        query: str,
        limit: int = 10,
        available_before: Optional[datetime] = None
    ) -> List[RetrievedChunk]:
        """Full-text search on chunk content."""
        sql = """
            SELECT 
                c.chunk_id, c.doc_id, c.content, c.section_path,
                c.chunk_index, c.start_offset, c.end_offset, c.token_count,
                ts_rank_cd(to_tsvector('jiebacfg', c.content), plainto_tsquery('jiebacfg', %s)) as score,
                d.available_at
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE to_tsvector('jiebacfg', c.content) @@ plainto_tsquery('jiebacfg', %s)
        """
        params = [query, query]
        
        if available_before:
            sql += " AND d.available_at <= %s"
            params.append(available_before)
        
        sql += " ORDER BY score DESC LIMIT %s"
        params.append(limit)
        
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        
        return [
            RetrievedChunk(
                chunk_id=row[0], doc_id=row[1], content=row[2],
                section_path=row[3], chunk_index=row[4],
                start_offset=row[5], end_offset=row[6], token_count=row[7],
                score=row[8], available_at=row[9]
            )
            for row in rows
        ]
    
    def vector_search(
        self,
        embedding: List[float],
        limit: int = 10,
        available_before: Optional[datetime] = None
    ) -> List[RetrievedChunk]:
        """Vector similarity search."""
        sql = """
            SELECT 
                c.chunk_id, c.doc_id, c.content, c.section_path,
                c.chunk_index, c.start_offset, c.end_offset, c.token_count,
                1 - (c.embedding <=> %s::vector) as score,
                d.available_at
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.embedding IS NOT NULL
        """
        params = [embedding]
        
        if available_before:
            sql += " AND d.available_at <= %s"
            params.append(available_before)
        
        sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
        params.extend([embedding, limit])
        
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        
        return [
            RetrievedChunk(
                chunk_id=row[0], doc_id=row[1], content=row[2],
                section_path=row[3], chunk_index=row[4],
                start_offset=row[5], end_offset=row[6], token_count=row[7],
                score=row[8], available_at=row[9]
            )
            for row in rows
        ]
    
    def hybrid_search(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        limit: int = 10,
        available_before: Optional[datetime] = None,
        fts_weight: float = 0.7,
        vector_weight: float = 0.3
    ) -> List[RetrievedChunk]:
        """Hybrid search using RRF (Reciprocal Rank Fusion)."""
        # Get full-text search results
        fts_results = self.full_text_search(
            query, limit=limit * 2, available_before=available_before
        )
        
        # Get vector search results if embedding provided
        vector_results = []
        if embedding:
            vector_results = self.vector_search(
                embedding, limit=limit * 2, available_before=available_before
            )
        
        # Combine using RRF
        k = 60  # RRF constant
        scores = {}
        chunk_map = {}
        
        # Score FTS results
        for rank, chunk in enumerate(fts_results):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + fts_weight / (k + rank + 1)
        
        # Score vector results
        for rank, chunk in enumerate(vector_results):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + vector_weight / (k + rank + 1)
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [
            RetrievedChunk(
                chunk_id=chunk_map[cid].chunk_id,
                doc_id=chunk_map[cid].doc_id,
                content=chunk_map[cid].content,
                section_path=chunk_map[cid].section_path,
                chunk_index=chunk_map[cid].chunk_index,
                start_offset=chunk_map[cid].start_offset,
                end_offset=chunk_map[cid].end_offset,
                token_count=chunk_map[cid].token_count,
                score=scores[cid],
                available_at=chunk_map[cid].available_at
            )
            for cid in sorted_ids[:limit]
        ]
