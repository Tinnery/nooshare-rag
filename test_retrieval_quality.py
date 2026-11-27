"""Test retrieval quality with synthetic corpus and ground truth."""
import psycopg2
from datetime import datetime
from document import Document, DocumentMetadata, DocumentType
from chunker import SectionChunker
from storage import ChunkRepository
from embedding import MockEmbeddingModel
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

# Ground truth: (query, expected_doc_ids, expected_keywords)
GROUND_TRUTH = [
    {
        "query": "000001.SZ 2024年新能源业务收入是多少？",
        "expected_doc_id": "test_000001_2024_annual",
        "expected_keywords": ["50亿元", "新能源业务"],
        "description": "Should find 2024 annual report for 000001.SZ with revenue 50亿元"
    },
    {
        "query": "000001.SZ 2023年研发投入占比",
        "expected_doc_id": "test_000001_2023_annual",
        "expected_keywords": ["5.88%", "研发投入"],
        "description": "Should find 2023 annual report with R&D ratio 5.88%"
    },
    {
        "query": "600000.SH 2024年净利润",
        "expected_doc_id": "test_600000_2024_annual",
        "expected_keywords": ["18亿元", "净利润"],
        "description": "Should find 600000.SH 2024 annual report with net profit 18亿元"
    },
    {
        "query": "000001.SZ 2024年传统业务收入",
        "expected_doc_id": "test_000001_2024_annual",
        "expected_keywords": ["45亿元", "传统业务"],
        "description": "Should find 2024 traditional business revenue 45亿元"
    }
]

def setup_test_data(conn):
    """Load synthetic corpus into database."""
    repo = ChunkRepository(conn)
    chunker = SectionChunker(max_chunk_size=500, overlap=50)
    embedding_model = MockEmbeddingModel(dim=1536)
    
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
    
    print(f"Loaded {len(SYNTHETIC_CORPUS)} documents, {len(all_chunks)} chunks")
    return repo, embedding_model

def test_retrieval_quality(repo, embedding_model):
    """Test retrieval against ground truth."""
    print("\n=== Retrieval Quality Test ===\n")
    
    passed = 0
    failed = 0
    
    for i, gt in enumerate(GROUND_TRUTH, 1):
        query = gt["query"]
        expected_doc_id = gt["expected_doc_id"]
        expected_keywords = gt["expected_keywords"]
        
        print(f"Test {i}: {query}")
        print(f"  Expected: {gt['description']}")
        
        # Vector search
        query_embedding = embedding_model.embed([query])[0]
        results = repo.vector_search(query_embedding, limit=5)
        
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
    return passed, failed

def test_pit_filtering(repo, embedding_model):
    """Test point-in-time filtering."""
    print("\n=== PIT Filtering Test ===\n")
    
    query = "000001.SZ 2024年新能源业务收入"
    query_embedding = embedding_model.embed([query])[0]
    
    # Test 1: Query at 2024-03-25 (before 2024 report available)
    print("Test 1: Query at 2024-03-25 (should only see 2023 report)")
    results = repo.vector_search(
        query_embedding,
        limit=5,
        available_before=datetime(2024, 3, 25)
    )
    
    if results:
        doc_ids = set(r.doc_id for r in results)
        if "test_000001_2023_annual" in doc_ids and "test_000001_2024_annual" not in doc_ids:
            print(f"  ✅ PASSED: Only 2023 report visible")
        else:
            print(f"  ❌ FAILED: Unexpected docs visible: {doc_ids}")
    else:
        print(f"  ❌ FAILED: No results")
    
    # Test 2: Query at 2025-03-25 (both 2023 and 2024 reports available)
    print("\nTest 2: Query at 2025-03-25 (should see both 2023 and 2024 reports)")
    results = repo.vector_search(
        query_embedding,
        limit=5,
        available_before=datetime(2025, 3, 25)
    )
    
    if results:
        doc_ids = set(r.doc_id for r in results)
        if "test_000001_2023_annual" in doc_ids and "test_000001_2024_annual" in doc_ids:
            print(f"  ✅ PASSED: Both reports visible")
        else:
            print(f"  ❌ FAILED: Missing expected docs: {doc_ids}")
    else:
        print(f"  ❌ FAILED: No results")

def main():
    conn = psycopg2.connect("postgresql://travis@localhost:5432/nooshare_dev")
    
    try:
        # Setup test data
        print("Setting up test data...")
        repo, embedding_model = setup_test_data(conn)
        
        # Run tests
        passed, failed = test_retrieval_quality(repo, embedding_model)
        test_pit_filtering(repo, embedding_model)
        
        print("\n=== Test Suite Complete ===")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()
