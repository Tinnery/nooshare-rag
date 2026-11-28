"""Database schema migrations for RAG storage."""
import psycopg2
from typing import Optional

# Default embedding dimension (OpenAI ada-002)
EMBEDDING_DIM = 1024


def migrate_v1(conn: psycopg2.extensions.connection) -> None:
    """Create initial schema for document chunks and embeddings."""
    
    with conn.cursor() as cur:
        # Enable required extensions
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # pg_trgm not available, removed
        
        # Documents table (metadata only, content stored in chunks)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                fiscal_period INTEGER NOT NULL,
                published_at TIMESTAMPTZ NOT NULL,
                available_at TIMESTAMPTZ NOT NULL,
                source_path TEXT NOT NULL,
                checksum TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Chunks table with vector embeddings
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                section_path TEXT NOT NULL DEFAULT '',
                chunk_index INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                embedding vector({EMBEDDING_DIM}),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Indexes for efficient retrieval
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id 
            ON chunks(doc_id);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_symbol_year 
            ON documents(symbol, fiscal_year, fiscal_period);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_available_at 
            ON documents(available_at);
        """)
        
        # Full-text search index on chunks
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_content_gin 
            ON chunks USING gin(to_tsvector('jiebacfg', content));
        """)
        
        # Trigram index skipped (pg_trgm not installed)
        
        conn.commit()


def migrate_v2_create_vector_index(conn: psycopg2.extensions.connection, 
                                    use_hnsw: bool = True) -> None:
    """Create vector similarity index (HNSW or IVFFlat)."""
    
    with conn.cursor() as cur:
        if use_hnsw:
            # HNSW index: better recall, slower build, more memory
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
                ON chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
        else:
            # IVFFlat index: faster build, needs training data
            # Note: requires at least 1 row to create index
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;")
            count = cur.fetchone()[0]
            
            if count > 0:
                n_lists = min(100, count // 10)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat 
                    ON chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {n_lists});
                """)
        
        conn.commit()


def get_migration_version(conn: psycopg2.extensions.connection) -> int:
    """Check current migration version."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chunks'
            );
        """)
        has_chunks = cur.fetchone()[0]
        
        if not has_chunks:
            return 0
        
        # Check if vector index exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM pg_indexes 
                WHERE indexname IN ('idx_chunks_embedding_hnsw', 'idx_chunks_embedding_ivfflat')
            );
        """)
        has_vector_index = cur.fetchone()[0]
        
        return 2 if has_vector_index else 1


def run_migrations(conn: psycopg2.extensions.connection) -> None:
    """Run all pending migrations."""
    version = get_migration_version(conn)
    
    if version < 1:
        print("Running migration v1: initial schema...")
        migrate_v1(conn)
        print("Migration v1 complete.")
    
    if version < 2:
        print("Running migration v2: vector index...")
        migrate_v2_create_vector_index(conn, use_hnsw=True)
        print("Migration v2 complete.")
    
    print(f"Current schema version: {get_migration_version(conn)}")


if __name__ == "__main__":
    # Test migration
    import os
    db_url = os.environ.get('DATABASE_URL', 'postgresql://travis@localhost:5432/nooshare_dev')
    
    conn = psycopg2.connect(db_url)
    try:
        run_migrations(conn)
    finally:
        conn.close()
