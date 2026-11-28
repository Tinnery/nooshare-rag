"""Test retrieval quality with real annual reports."""
import sys
import os
from datetime import datetime
from pathlib import Path

# Add nooshare-rag to path
sys.path.insert(0, '/home/travis/nooshare-workenv/repos/nooshare-rag')

from document import Document, DocumentMetadata, DocumentType
from chunker import SectionChunker
from storage import ChunkRepository
from embedding import OllamaEmbeddingModel
import psycopg2

# Real annual report files
ANNUAL_REPORTS = [
    {
        "path": "/home/travis/nooshare-workenv/repos/nooshare-annual-semi-extraction/data/md_samples/000001_2024_4.md",
        "doc_id": "real_000001_2024_annual",
        "symbol": "000001.SZ",
        "fiscal_year": 2024,
        "fiscal_period": 4,
        "published_at": datetime(2025, 3, 15),
        "available_at": datetime(2025, 3, 20),
    },
    {
        "path": "/home/travis/nooshare-workenv/repos/nooshare-annual-semi-extraction/data/md_samples/000001_2023_4.md",
        "doc_id": "real_000001_2023_annual",
        "symbol": "000001.SZ",
        "fiscal_year": 2023,
        "fiscal_period": 4,
        "published_at": datetime(2024, 3, 15),
        "available_at": datetime(2024, 3, 20),
    },
    {
        "path": "/home/travis/nooshare-workenv/repos/nooshare-annual-semi-extraction/data/md_samples/600519_2023_4.md",
        "doc_id": "real_600519_2023_annual",
        "symbol": "600519.SH",
        "fiscal_year": 2023,
        "fiscal_period": 4,
        "published_at": datetime(2024, 3, 20),
        "available_at": datetime(2024, 3, 25),
    },
]

# Ground truth queries
GROUND_TRUTH = [
    {
        "query": "平安银行2024年净利润是多少？",
        "expected_doc_id": "real_000001_2024_annual",
        "expected_keywords": ["445.08", "净利润"],
        "description": "Should find 000001 2024 annual report with net profit 445.08 billion"
    },
    {
        "query": "平安银行2024年资产总额",
        "expected_doc_id": "real_000001_2024_annual",
        "expected_keywords": ["57,692.70", "资产总额"],
        "description": "Should find 000001 2024 annual report with total assets 57,692.70 billion"
    },
    {
        "query": "贵州茅台2023年营业收入",
        "expected_doc_id": "real_600519_2023_annual",
        "expected_keywords": ["1,476.94", "营业收入"],
        "description": "Should find 600519 2023 annual report with revenue 1,476.94 billion"
    },
    {
        "query": "贵州茅台2023年基本每股收益",
        "expected_doc_id": "real_600519_2023_annual",
        "expected_keywords": ["59.49", "基本每股收益"],
        "description": "Should find 600519 2023 annual report with EPS 59.49"
    },
    {
        "query": "平安银行2023年分红方案",
        "expected_doc_id": "real_000001_2023_annual",
        "expected_keywords": ["7.19", "现金股利"],
        "description": "Should find 000001 2023 annual report with dividend 7.19 per 10 shares"
    },
]

def load_real_annual_reports():
    """Load real annual reports into database."""
    conn = psycopg2.connect("postgresql://travis@localhost:5432/nooshare_dev")
    repo = ChunkRepository(conn)
    embedding_model = OllamaEmbeddingModel(model="qwen3-embedding:0.6b")
    chunker = SectionChunker(max_chunk_size=2000, overlap=300)
    
    total_chunks = 0
    
    for report in ANNUAL_REPORTS:
        print(f"Loading {report['doc_id']}...")
        
        # Read file content
        with open(report['path'], 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create document
        metadata = DocumentMetadata(
            doc_id=report['doc_id'],
            doc_type=DocumentType.ANNUAL_REPORT,
            symbol=report['symbol'],
            fiscal_year=report['fiscal_year'],
            fiscal_period=report['fiscal_period'],
            published_at=report['published_at'],
            available_at=report['available_at'],
            source_path=report['path'],
        )
        doc = Document(metadata=metadata, content=content)
        
        # Save document
        repo.save_document(doc)
        
        # Chunk document
        chunks = chunker.chunk(doc)
        total_chunks += len(chunks)
        
        # Save chunks
        repo.save_chunks(chunks)
        
        # Generate and save embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = embedding_model.embed(chunk_texts)
        chunk_ids = [c.chunk_id for c in chunks]
        repo.save_embeddings(chunk_ids, embeddings)
        
        print(f"  Loaded {len(chunks)} chunks")
    
    conn.commit()
    conn.close()
    
    print(f"\nTotal loaded: {len(ANNUAL_REPORTS)} documents, {total_chunks} chunks")

def test_retrieval():
    """Test retrieval against ground truth."""
    conn = psycopg2.connect("postgresql://travis@localhost:5432/nooshare_dev")
    repo = ChunkRepository(conn)
    embedding_model = OllamaEmbeddingModel(model="qwen3-embedding:0.6b")
    
    print("\n=== Real Annual Report Retrieal Test ===\n")
    
    passed = 0
    failed = 0
    
    for i, gt in enumerate(GROUND_TRUTH, 1):
        query = gt["query"]
        expected_doc_id = gt["expected_doc_id"]
        expected_keywords = gt["expected_keywords"]
        
        print(f"Test {i}: {query}")
        print(f"  Expected: {gt['description']}")
        
        # Hybrid search (FTS + vector)
        query_embedding = embedding_model.embed([query])[0]
        results = repo.hybrid_search(query, query_embedding, limit=10)
        
        if not results:
            print(f"  ❌ FAILED: No results returned")
            failed += 1
            continue
        
        # Check if top result is from expected document
        top_result = results[0]
        if top_result.doc_id == expected_doc_id:
            # Check if expected keywords are in content
            content = top_result.content
            keywords_found = [kw for kw in expected_keywords if kw in content]
            
            if len(keywords_found) == len(expected_keywords):
                print(f"  ✅ PASSED: Top result is {top_result.doc_id}")
                print(f"     Score: {top_result.score:.4f}")
                print(f"     Keywords found: {keywords_found}")
                passed += 1
            else:
                print(f"  ⚠️  PARTIAL: Correct doc but missing keywords")
                print(f"     Found: {keywords_found}, Missing: {set(expected_keywords) - set(keywords_found)}")
                failed += 1
        else:
            print(f"  ❌ FAILED: Top result is {top_result.doc_id}, expected {expected_doc_id}")
            print(f"     Score: {top_result.score:.4f}")
            failed += 1
        
        print()
    
    print(f"=== Results: {passed}/{len(GROUND_TRUTH)} passed, {failed} failed ===")
    
    conn.close()
    return passed > 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        load_real_annual_reports()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        success = test_retrieval()
        sys.exit(0 if success else 1)
    else:
        print("Usage: python test_real_annual_report.py [load|test]")
        sys.exit(1)
