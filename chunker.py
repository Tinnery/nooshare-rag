"""Text chunking with section and position tracking."""
from dataclasses import dataclass
from typing import List, Optional
import re
from document import Document


@dataclass
class Chunk:
    """A text chunk with provenance."""
    chunk_id: str
    doc_id: str
    content: str
    section_path: str  # e.g., "第二节 董事会报告 > 一、经营情况讨论与分析"
    chunk_index: int  # Position within document
    start_offset: int  # Character offset in original
    end_offset: int
    token_count: int  # Approximate token count


class SectionChunker:
    """Chunk documents by section headings, preserving hierarchy."""
    
    # Chinese section heading patterns
    HEADING_PATTERNS = [
        (1, r'^(第[一二三四五六七八九十百千]+[节章部])\s+(.+)$'),  # 第一节 董事会报告
        (2, r'^([一二三四五六七八九十百千]+)[、.．]\s*(.+)$'),  # 一、经营情况
        (3, r'^\(([一二三四五六七八九十]+)\)\s*(.+)$'),  # (一) 基本情况
        (4, r'^(\d+)[、.．]\s*(.+)$'),  # 1、具体事项
        (5, r'^\((\d+)\)\s*(.+)$'),  # (1) 详细说明
    ]
    
    def __init__(self, max_chunk_size: int = 2000, overlap: int = 300):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk(self, doc: Document) -> List[Chunk]:
        """Split document into chunks by section."""
        lines = doc.content.split('\n')
        sections = self._extract_sections(lines)
        chunks = []
        chunk_idx = 0
        
        # Build metadata prefix for FTS matching
        metadata_prefix = self._build_metadata_prefix(doc)
        
        for section in sections:
            section_chunks = self._split_section(section, doc.doc_id, chunk_idx, metadata_prefix)
            chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)
        
        return chunks
    
    def _extract_sections(self, lines: List[str]) -> List[dict]:
        """Extract sections with hierarchy from lines."""
        sections = []
        current_section = {
            'path': '',
            'level': 0,
            'content': [],
            'start_offset': 0
        }
        offset = 0
        path_stack = []  # [(level, title), ...]
        
        for line in lines:
            line_stripped = line.strip()
            heading = self._match_heading(line_stripped)
            
            if heading:
                # Save current section
                if current_section['content']:
                    sections.append(current_section)
                
                level, title = heading
                # Update path stack
                path_stack = [(l, t) for l, t in path_stack if l < level]
                path_stack.append((level, title))
                path = ' > '.join(t for _, t in path_stack)
                
                current_section = {
                    'path': path,
                    'level': level,
                    'content': [],
                    'start_offset': offset
                }
            
            current_section['content'].append(line)
            offset += len(line) + 1  # +1 for newline
        
        # Save last section
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _match_heading(self, line: str) -> Optional[tuple]:
        """Match a heading pattern, return (level, title) or None."""
        for level, pattern in self.HEADING_PATTERNS:
            match = re.match(pattern, line)
            if match:
                return (level, line)
        return None
    
    def _build_metadata_prefix(self, doc: Document) -> str:
        """Build a metadata prefix to prepend to each chunk for better FTS matching."""
        symbol = doc.metadata.symbol
        fiscal_year = doc.metadata.fiscal_year
        fiscal_period = doc.metadata.fiscal_period
        
        # Common company name mapping
        company_names = {
            '000001.SZ': '平安银行',
            '600519.SH': '贵州茅台',
        }
        
        company_name = company_names.get(symbol, symbol)
        period_names = {1: '一季报', 2: '半年报', 3: '三季报', 4: '年报'}
        period_name = period_names.get(fiscal_period, '')
        
        return f'{company_name} {fiscal_year}年{period_name} '

    
    def _split_section(self, section: dict, doc_id: str, start_idx: int, metadata_prefix: str = '') -> List[Chunk]:
        """Split a section into chunks if too large."""
        content = '\n'.join(section['content'])
        chunks = []
        
        if len(content) <= self.max_chunk_size:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{start_idx}",
                doc_id=doc_id,
                content=metadata_prefix + content,
                section_path=section['path'],
                chunk_index=start_idx,
                start_offset=section['start_offset'],
                end_offset=section['start_offset'] + len(content),
                token_count=len(content) // 2  # Rough estimate for Chinese
            ))
        else:
            # Split large sections with overlap
            pos = 0
            idx = start_idx
            while pos < len(content):
                end = min(pos + self.max_chunk_size, len(content))
                chunk_content = content[pos:end]
                
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    content=metadata_prefix + chunk_content,
                    section_path=section['path'],
                    chunk_index=idx,
                    start_offset=section['start_offset'] + pos,
                    end_offset=section['start_offset'] + end,
                    token_count=len(chunk_content) // 2
                ))
                
                pos = end - self.overlap if end < len(content) else end
                idx += 1
        
        return chunks
