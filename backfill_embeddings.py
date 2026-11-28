#!/usr/bin/env python3
"""Backfill embeddings for chunks that don't have them yet."""
import psycopg2
import requests
from typing import List

# Database config
DB_CONFIG = {
    "host": "localhost",
    "database": "nooshare_dev",
    "user": "travis"
}

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "qwen3-embedding:4b"
MAX_DIM = 2000  # Match vector(2000) column
BATCH_SIZE = 100


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_pending_chunks(limit: int = BATCH_SIZE) -> List[tuple]:
    """Get chunks without embeddings."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_id, content 
        FROM chunks 
        WHERE embedding IS NULL 
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Create embeddings using Ollama batch API."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=300
    )
    response.raise_for_status()
    result = response.json()
    embeddings = result["embeddings"]
    # Truncate to MAX_DIM
    return [emb[:MAX_DIM] for emb in embeddings]


def save_embeddings_batch(chunk_ids: List[str], embeddings: List[List[float]]) -> None:
    """Save embeddings to database."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    for chunk_id, embedding in zip(chunk_ids, embeddings):
        # Convert embedding to pgvector format
        emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
        cur.execute("""
            UPDATE chunks 
            SET embedding = %s::vector
            WHERE chunk_id = %s
        """, (emb_str, chunk_id))
    
    conn.commit()
    cur.close()
    conn.close()


def main():
    """Main backfill loop."""
    total_processed = 0
    total_failed = 0
    
    print("Starting embedding backfill...")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max dimension: {MAX_DIM}")
    
    while True:
        # Get pending chunks
        chunks = get_pending_chunks(BATCH_SIZE)
        if not chunks:
            print("No more chunks to process.")
            break
        
        chunk_ids = [c[0] for c in chunks]
        contents = [c[1] for c in chunks]
        
        try:
            # Create embeddings batch
            embeddings = create_embeddings_batch(contents)
            
            # Save to database
            save_embeddings_batch(chunk_ids, embeddings)
            
            total_processed += len(chunk_ids)
            print(f"Processed batch of {len(chunk_ids)} chunks. Total: {total_processed}")
            
        except Exception as e:
            print(f"Failed batch: {e}")
            total_failed += len(chunk_ids)
    
    print(f"\nBackfill complete!")
    print(f"  Processed: {total_processed}")
    print(f"  Failed: {total_failed}")


if __name__ == "__main__":
    main()
