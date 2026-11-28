"""Pytest fixtures for RAG tests."""
import pytest
import psycopg2
from datetime import datetime
from document import Document, DocumentMetadata, DocumentType
from chunker import SectionChunker
from storage import ChunkRepository
from embedding import OllamaEmbeddingModel
from migrations import run_migrations

# Synthetic corpus with known facts
SYNTHETIC_CORPUS = [
    {
        "doc_id": "test_000001_2023_annual",
        "symbol": "000001.SZ",
        "fiscal_year": 2023,
        "fiscal_period": 4,
        "published_at": datetime(2024, 3, 15),
        "available_at": datetime(2024, 3, 20),
        "content": """
第一节 董事会报告

一、2023年度经营情况

2023年度，公司实现营业收入85亿元，同比增长10%。净利润12亿元，同比增长8%。

（一）主营业务收入

主营业务收入80亿元，其中：
1、新能源业务收入35亿元，同比增长25%。
2、传统业务收入45亿元，同比增长5%。

（二）研发投入

2023年研发投入5亿元，占营业收入比例5.88%。
"""
    },
    {
        "doc_id": "test_000001_2024_annual",
        "symbol": "000001.SZ",
        "fiscal_year": 2024,
        "fiscal_period": 4,
        "published_at": datetime(2025, 3, 15),
        "available_at": datetime(2025, 3, 20),
        "content": """
第一节 董事会报告

一、2024年度经营情况

2024年度，公司实现营业收入100亿元，同比增长17.65%。净利润15亿元，同比增长25%。

（一）主营业务收入

主营业务收入95亿元，其中：
1、新能源业务收入50亿元，同比增长42.86%。
2、传统业务收入45亿元，与上年持平。

（二）研发投入

2024年研发投入7亿元，占营业收入比例7%。
"""
    },
    {
        "doc_id": "test_600000_2024_annual",
        "symbol": "600000.SH",
        "fiscal_year": 2024,
        "fiscal_period": 4,
        "published_at": datetime(2025, 4, 10),
        "available_at": datetime(2025, 4, 15),
        "content": """
第一节 董事会报告

一、2024年度经营情况

2024年度，公司实现营业收入120亿元，同比增长20%。净利润18亿元，同比增长15%。

（一）主营业务收入

主营业务收入115亿元，其中：
1、新能源业务收入60亿元，同比增长50%。
2、传统业务收入55亿元，同比增长5%。

（二）研发投入

2024年研发投入8亿元，占营业收入比例6.67%。
"""
    }
]

@pytest.fixture(scope="session")
def db_connection():
    """Create database connection for tests."""
    conn = psycopg2.connect("postgresql://travis@localhost:5432/nooshare_dev")
    
    # Run migrations
    run_migrations(conn)
    conn.commit()
    
    yield conn
    
    # Cleanup: delete test data
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE doc_id LIKE 'test_%'")
    cur.execute("DELETE FROM documents WHERE doc_id LIKE 'test_%'")
    conn.commit()
    cur.close()
    conn.close()

@pytest.fixture(scope="session")
def repo(db_connection):
    """Create ChunkRepository instance."""
    return ChunkRepository(db_connection)

@pytest.fixture(scope="session")
def embedding_model():
    """Create OllamaEmbeddingModel instance."""
    return OllamaEmbeddingModel(model="qwen3-embedding:4b")

@pytest.fixture(scope="session", autouse=True)
def setup_test_data(repo, embedding_model):
    """Load synthetic corpus into database."""
    chunker = SectionChunker(max_chunk_size=500, overlap=50)
    
    all_chunks = []
    all_texts = []
    
    for doc_data in SYNTHETIC_CORPUS:
        metadata = DocumentMetadata(
            doc_id=doc_data["doc_id"],
            doc_type=DocumentType.ANNUAL_REPORT,
            symbol=doc_data["symbol"],
            fiscal_year=doc_data["fiscal_year"],
            fiscal_period=doc_data["fiscal_period"],
            published_at=doc_data["published_at"],
            available_at=doc_data["available_at"],
            source_path=f"/test/{doc_data['doc_id']}.md"
        )
        doc = Document(metadata=metadata, content=doc_data["content"])
        
        repo.save_document(doc)
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        all_texts.extend([c.content for c in chunks])
    
    repo.save_chunks(all_chunks)
    
    # Generate and save embeddings
    embeddings = embedding_model.embed(all_texts)
    chunk_ids = [c.chunk_id for c in all_chunks]
    repo.save_embeddings(chunk_ids, embeddings)
    
    print(f"\nLoaded {len(SYNTHETIC_CORPUS)} documents, {len(all_chunks)} chunks")
