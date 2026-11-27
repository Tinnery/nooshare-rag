"""Test retrieval quality with synthetic corpus and ground truth."""
from datetime import datetime

# Ground truth: (query, expected_doc_id, expected_keywords)
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
    assert passed > 0, f"All retrieval tests failed ({passed}/{len(GROUND_TRUTH)})"


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
    
    assert results, "No results returned for PIT test 1"
    doc_ids = set(r.doc_id for r in results)
    assert "test_000001_2023_annual" in doc_ids, f"Missing 2023 report: {doc_ids}"
    assert "test_000001_2024_annual" not in doc_ids, f"2024 report should not be visible: {doc_ids}"
    print(f"  ✅ PASSED: Only 2023 report visible")
    
    # Test 2: Query at 2025-03-25 (both 2023 and 2024 reports available)
    print("\nTest 2: Query at 2025-03-25 (should see both 2023 and 2024 reports)")
    results = repo.vector_search(
        query_embedding,
        limit=5,
        available_before=datetime(2025, 3, 25)
    )
    
    assert results, "No results returned for PIT test 2"
    doc_ids = set(r.doc_id for r in results)
    assert "test_000001_2023_annual" in doc_ids, f"Missing 2023 report: {doc_ids}"
    assert "test_000001_2024_annual" in doc_ids, f"Missing 2024 report: {doc_ids}"
    print(f"  ✅ PASSED: Both reports visible")
