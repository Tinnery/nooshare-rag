"""Test chunker with synthetic document."""
from datetime import datetime
from document import Document, DocumentMetadata, DocumentType
from chunker import SectionChunker

# Synthetic test document
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

# Print results
print(f"Document: {doc.doc_id}")
print(f"Total chunks: {len(chunks)}")
print()

for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.section_path}")
    print(f"  Offset: {chunk.start_offset}-{chunk.end_offset}")
    print(f"  Tokens: ~{chunk.token_count}")
    print(f"  Content preview: {chunk.content[:80]}...")
    print()

print("Test passed!")
