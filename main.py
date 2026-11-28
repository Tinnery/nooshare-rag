import logging
import time
from datetime import datetime
from typing import List, Optional

import psycopg2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from retrieval_service import RetrievalService

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NooShare RAG API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
conn = psycopg2.connect("postgresql://travis@localhost/nooshare_dev")
retrieval_service = RetrievalService(conn)


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    rerank: bool = False
    available_before: Optional[str] = None


class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    section_path: str
    chunk_index: int
    start_offset: int
    end_offset: int
    token_count: int
    available_at: Optional[str]
    score: float


class DocumentInfo(BaseModel):
    doc_id: str
    doc_type: str
    symbol: str
    fiscal_year: int
    fiscal_period: int
    source_path: str
    created_at: Optional[str]
    available_at: Optional[str]


# Middleware for request logging and performance monitoring
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        "%s %s | status=%s | duration=%.3fs",
        request.method, request.url.path, response.status_code, process_time
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
def health():
    logger.debug("Health check requested")
    return {"status": "ok"}


@app.post("/search", response_model=List[ChunkResponse])
def search_endpoint(request: SearchRequest):
    logger.info(
        "Search request | query='%s...' | limit=%d | rerank=%s",
        request.query[:50], request.limit, request.rerank
    )

    try:
        start_time = time.time()

        available_before = None
        if request.available_before:
            available_before = datetime.fromisoformat(request.available_before)

        results = retrieval_service.search(
            query=request.query,
            limit=request.limit,
            rerank=request.rerank,
            available_before=available_before
        )

        response = []
        for r in results:
            response.append(ChunkResponse(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                content=r.content,
                section_path=r.section_path,
                chunk_index=r.chunk_index,
                start_offset=r.start_offset,
                end_offset=r.end_offset,
                token_count=r.token_count,
                available_at=r.available_at.isoformat() if r.available_at else None,
                score=r.score
            ))

        search_time = time.time() - start_time
        logger.info(
            "Search completed | results=%d | duration=%.3fs",
            len(response), search_time
        )

        return response
    except Exception as e:
        logger.error("Search failed | error=%s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents(limit: int = Query(default=100, ge=1, le=1000), offset: int = Query(default=0, ge=0)):
    logger.info("List documents | limit=%d | offset=%d", limit, offset)

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT doc_id, doc_type, symbol, fiscal_year, fiscal_period, source_path, created_at, available_at
                FROM documents
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            rows = cur.fetchall()

        response = []
        for row in rows:
            response.append({
                "doc_id": row[0],
                "doc_type": row[1],
                "symbol": row[2],
                "fiscal_year": row[3],
                "fiscal_period": row[4],
                "source_path": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
                "available_at": row[7].isoformat() if row[7] else None
            })

        logger.info("Documents listed | count=%d", len(response))
        return response
    except Exception as e:
        logger.error("List documents failed | error=%s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    logger.info("Stats requested")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*), COUNT(embedding) FROM chunks")
            chunk_count, embedded_count = cur.fetchone()

        result = {
            "documents": doc_count,
            "chunks": chunk_count,
            "chunks_with_embedding": embedded_count,
            "chunks_pending_embedding": chunk_count - embedded_count
        }

        logger.info(
            "Stats returned | documents=%d | chunks=%d | embedded=%d",
            doc_count, chunk_count, embedded_count
        )
        return result
    except Exception as e:
        logger.error("Stats failed | error=%s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting NooShare RAG API server on port 8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
