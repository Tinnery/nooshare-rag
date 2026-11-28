#!/usr/bin/env python3
"""Batch load annual reports from RustFS with performance optimization.

Features:
- Parallel download with ThreadPoolExecutor
- Async embedding with concurrent API calls
- Progress tracking and error handling
- Configurable batch sizes and concurrency

Usage:
    # Download with parallel processing
    python load_annual_reports.py download --years 2024-2024 --limit 100 --workers 8
    
    # Generate embeddings with async API calls
    python load_annual_reports.py embed --batch-size 32 --workers 4
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Any

import boto3
from botocore.config import Config
import psycopg2

from document import Document, DocumentMetadata, DocumentType
from chunker import SectionChunker
from storage import ChunkRepository

# RustFS config
RUSTFS_ENDPOINT = "http://127.0.0.1:9000"
RUSTFS_ACCESS_KEY = "charlie"
RUSTFS_SECRET_KEY = "sheeta-rustfs"
RUSTFS_BUCKET = "annual-semi"

# Database config
DB_DSN = "postgresql://travis@localhost/nooshare_dev"

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "qwen3-embedding:4b"


def parse_filename(key: str) -> Tuple[str, int, int]:
    """Parse stock code, year, period from S3 key."""
    match = re.search(r'published/(\d+)/(\d+)_(\d{4})_(\d)\.md', key)
    if not match:
        raise ValueError(f"Cannot parse key: {key}")
    stock_code = match.group(1)
    year = int(match.group(3))
    period = int(match.group(4))
    return stock_code, year, period


def list_reports(
    client: boto3.client,
    years: Tuple[int, int] = None,
    limit: int = None,
) -> Iterator[str]:
    """List report keys from RustFS."""
    paginator = client.get_paginator('list_objects_v2')
    count = 0
    for page in paginator.paginate(Bucket=RUSTFS_BUCKET, Prefix='published/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if years:
                try:
                    _, year, _ = parse_filename(key)
                    if not (years[0] <= year <= years[1]):
                        continue
                except ValueError:
                    continue
            yield key
            count += 1
            if limit and count >= limit:
                return


def download_report(client: boto3.client, key: str) -> str:
    """Download report content from RustFS."""
    response = client.get_object(Bucket=RUSTFS_BUCKET, Key=key)
    return response['Body'].read().decode('utf-8')


def create_document(key: str, content: str) -> Document:
    """Create Document from S3 key and content."""
    stock_code, year, period = parse_filename(key)
    doc_type = DocumentType.ANNUAL_REPORT if period == 4 else DocumentType.SEMI_ANNUAL_REPORT
    doc_id = f"{stock_code}_{year}_{period}"
    published_at = datetime(year, 4, 30) if period == 4 else datetime(year, 8, 31)
    
    metadata = DocumentMetadata(
        doc_id=doc_id,
        doc_type=doc_type,
        symbol=f"{stock_code}.SZ" if stock_code.startswith('0') else f"{stock_code}.SH",
        fiscal_year=year,
        fiscal_period=period,
        published_at=published_at,
        available_at=published_at,
        source_path=key,
        checksum=hashlib.md5(content.encode()).hexdigest(),
    )
    
    return Document(metadata=metadata, content=content)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using Ollama API."""
    data = json.dumps({"model": EMBEDDING_MODEL, "input": texts}).encode('utf-8')
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
    embeddings = result["embeddings"]
    # Truncate to 2000 dims (pgvector index limit)
    return [e[:2000] for e in embeddings]


def process_single_document(
    key: str,
    client: boto3.client,
    chunker: SectionChunker
) -> Tuple[str, Document, List]:
    """Process a single document: download, parse, chunk."""
    content = download_report(client, key)
    doc = create_document(key, content)
    chunks = chunker.chunk(doc)
    return key, doc, chunks


def cmd_download(args):
    """Download and store documents with parallel processing."""
    years = None
    if args.years:
        start, end = args.years.split('-')
        years = (int(start), int(end))
    
    client = boto3.client(
        's3',
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=RUSTFS_ACCESS_KEY,
        aws_secret_access_key=RUSTFS_SECRET_KEY,
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}),
        verify=False,
    )
    
    keys = list(list_reports(client, years=years, limit=args.limit))
    print(f"Found {len(keys)} reports to download")
    
    if args.dry_run:
        for key in keys[:10]:
            print(f"  {key}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more")
        return
    
    conn = psycopg2.connect(DB_DSN)
    repo = ChunkRepository(conn)
    chunker = SectionChunker(max_chunk_size=2000, overlap=300)
    
    total_docs = 0
    total_chunks = 0
    errors = 0
    
    # Parallel download and processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_document, key, client, chunker): key
            for key in keys
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                key, doc, chunks = future.result()
                
                # Save to database
                repo.save_document(doc)
                repo.save_chunks(chunks)
                
                total_docs += 1
                total_chunks += len(chunks)
                
                if i % 10 == 0 or i == len(keys):
                    print(f"[{i}/{len(keys)}] Downloaded {total_docs} docs, {total_chunks} chunks")
                    conn.commit()
            
            except Exception as e:
                errors += 1
                print(f"Error processing {key}: {e}", file=sys.stderr)
                conn.rollback()
    
    conn.commit()
    print(f"\nCompleted: {total_docs} documents, {total_chunks} chunks, {errors} errors")
    conn.close()


def embed_batch(batch_ids: List[str], batch_texts: List[str]) -> Tuple[List[str], List[List[float]]]:
    """Embed a batch of texts."""
    embeddings = embed_texts(batch_texts)
    return batch_ids, embeddings


def cmd_embed(args):
    """Generate embeddings with async API calls."""
    conn = psycopg2.connect(DB_DSN)
    repo = ChunkRepository(conn)
    
    # Get chunks without embeddings
    cur = conn.cursor()
    cur.execute("""
        SELECT c.chunk_id, c.content
        FROM chunks c
        WHERE c.embedding IS NULL
        LIMIT %s
    """, (args.limit,))
    
    rows = cur.fetchall()
    print(f"Found {len(rows)} chunks to embed")
    
    if not rows:
        print("No chunks need embedding")
        return
    
    total = 0
    batches = []
    
    # Prepare batches
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        batch_ids = [row[0] for row in batch]
        batch_texts = [row[1] for row in batch]
        batches.append((batch_ids, batch_texts))
    
    print(f"Processing {len(batches)} batches with {args.workers} workers")
    
    # Parallel embedding
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(embed_batch, batch_ids, batch_texts): (batch_ids, batch_texts)
            for batch_ids, batch_texts in batches
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                batch_ids, embeddings = future.result()
                repo.save_embeddings(batch_ids, embeddings)
                conn.commit()
                
                total += len(batch_ids)
                print(f"[{i}/{len(batches)}] Embedded {total} chunks")
            
            except Exception as e:
                print(f"Error embedding batch: {e}", file=sys.stderr)
                conn.rollback()
    
    print(f"\nCompleted: {total} chunks embedded")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load annual reports from RustFS")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    dl_parser = subparsers.add_parser('download', help='Download and store documents')
    dl_parser.add_argument('--years', type=str, help='Year range, e.g., 2020-2024')
    dl_parser.add_argument('--limit', type=int, help='Max number of reports to download')
    dl_parser.add_argument('--dry-run', action='store_true', help='List files without downloading')
    dl_parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    
    # Embed command
    emb_parser = subparsers.add_parser('embed', help='Generate embeddings for stored documents')
    emb_parser.add_argument('--limit', type=int, default=10000, help='Max chunks to embed')
    emb_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding')
    emb_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        cmd_download(args)
    elif args.command == 'embed':
        cmd_embed(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
