#!/usr/bin/env python3
"""Test retrieval service with reranking."""
import psycopg2
from retrieval_service import create_retrieval_service

def test_search():
    """Test search with and without reranking."""
    service = create_retrieval_service()
    
    queries = [
        "平安银行2024年净利润",
        "贵州茅台营业收入",
        "总资产规模"
    ]
    
    print("=== Test 1: Search without reranking ===")
    for query in queries:
        results = service.search(query, limit=3, rerank=False)
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results")
        for i, r in enumerate(results, 1):
            print(f"  {i}. doc_id={r.doc_id} score={r.score:.4f}")
            print(f"     {r.content[:100]}...")
    
    print("\n\n=== Test 2: Search with reranking ===")
    for query in queries:
        results = service.search(query, limit=3, rerank=True, rerank_top_k=10)
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results")
        for i, r in enumerate(results, 1):
            print(f"  {i}. doc_id={r.doc_id} score={r.score:.4f}")
            print(f"     {r.content[:100]}...")

if __name__ == '__main__':
    test_search()
