"""Document input protocol for RAG ingestion."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class DocumentType(str, Enum):
    """Document type enumeration."""
    ANNUAL_REPORT = "annual_report"
    SEMI_ANNUAL_REPORT = "semi_annual_report"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    doc_id: str
    doc_type: DocumentType
    symbol: str  # e.g., "000001.SZ"
    fiscal_year: int
    fiscal_period: int  # 1=Q1, 2=H1, 3=Q3, 4=Annual
    published_at: datetime
    available_at: datetime  # PIT cutoff
    source_path: str
    checksum: Optional[str] = None


@dataclass
class Document:
    """A document ready for chunking."""
    metadata: DocumentMetadata
    content: str  # Full text content
    
    @property
    def doc_id(self) -> str:
        return self.metadata.doc_id
