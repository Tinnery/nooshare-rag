"""Test storage layer with synthetic data."""
import psycopg2
from datetime import datetime
from document import Document, DocumentMetadata, DocumentType
from chunker import SectionChunker
from migrations import run_migrations
from storage import ChunkRepository
from embedding import OllamaEmbeddingModel

# Test document
test_content = """
第一节 董事会报告

一、经营情况讨论与分析

2024年度，公司实现营业收入100亿元，同比增长15%。主要业务板块表现良好，其中新能源业务增长显著。

（一）主营业务收入

主营业务收入达到95亿元，占比95%。具体分业务如下：
1、新能源业务收入50亿元，同比增长30%。
2、传统业务收入45亿元，同比增长5%。

（二）成本费用分析

营业成本70亿元，管理费用10亿元，财务费用5亿元。

二、未来发展展望

2025年，公司将继续加大新能源领域投入，预计收入增长20%以上。

第二节 监事会报告

一、监事会工作情况

2024年监事会共召开4次会议，审议重大事项12项。
"""

# Create test document
metadata = DocumentMetadata(
    doc_id="test_000001_2024_annual",
    doc_type=DocumentType.ANNUAL_REPORT,
    symbol="000001.SZ",
    fiscal_year=2024,
    fiscal_period=4,
    published_at=datetime(2025, 3, 15),
    available_at=datetime(2025, 3, 20),
    source_path="/data/reports/000001/2024_annual.md"
)

doc = Document(metadata=metadata, content=test_content)

# Chunk the document
chunker = SectionChunker(max_chunk_size=500, overlap=50)
chunks = chunker.chunk(doc)

print(f"Document: {doc.doc_id}")
print(f"Total chunks: {len(chunks)}")

# Connect to database
conn = psycopg2.connect("postgresql://travis@localhost:5432/nooshare_dev")

try:
    # Run migrations
    print("\nRunning migrations...")
    run_migrations(conn)

    # Save document and chunks
    repo = ChunkRepository(conn)

    print("\nSaving document...")
    repo.save_document(doc)

    print("Saving chunks...")
    repo.save_chunks(chunks)

    # Generate and save embeddings
    print("\nGenerating embeddings...")
    embedding_model = OllamaEmbeddingModel(model="qwen3-embedding:8b")
    texts = [chunk.content for chunk in chunks]
    embeddings = embedding_model.embed(texts)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    print(f"Saving {len(embeddings)} embeddings...")
    repo.save_embeddings(chunk_ids, embeddings)

    # Test vector search
    print("\nTesting vector search for '新能源业务'...")
    query_embedding = embedding_model.embed(["新能源业务"])[0]
    vector_results = repo.vector_search(query_embedding, limit=3)
    print(f"Found {len(vector_results)} results:")
    for r in vector_results:
        print(f"  - {r.chunk_id}: score={r.score:.4f}, section={r.section_path}")
        print(f"    Content: {r.content[:60]}...")

    # Test hybrid search
    print("\nTesting hybrid search...")
    hybrid_results = repo.hybrid_search(
        query="新能源业务",
        embedding=query_embedding,
        limit=3
    )
    print(f"Found {len(hybrid_results)} results:")
    for r in hybrid_results:
        print(f"  - {r.chunk_id}: score={r.score:.4f}")

    # Test PIT filter
    print("\nTesting PIT filter (available_before=2025-03-19)...")
    pit_results = repo.vector_search(
        query_embedding,
        limit=3,
        available_before=datetime(2025, 3, 19)
    )
    print(f"Found {len(pit_results)} results (should be 0, doc available_at=2025-03-20)")

    print("\nTest passed!")

finally:
    conn.close()
