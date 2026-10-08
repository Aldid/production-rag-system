"""Recursive semantic text chunker with heading awareness and token-bounded overlap."""

import re
from typing import List, Optional
from src.ingestion.metadata import Chunk, Document


class RecursiveSemanticChunker:
    """Chunks documents recursively along structural boundaries:
    1. Markdown Headings (# , ## , ### )
    2. Double Newlines (Paragraphs)
    3. Sentence Boundaries (. , ! , ? )
    4. Words (Whitespace)
    """

    def __init__(self, target_chunk_size: int = 400, overlap: int = 80, min_chunk_size: int = 50):
        self.target_size = target_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.heading_pattern = re.compile(r"^(#{1,4}\s+.+)$", re.MULTILINE)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimation (~4 characters per token)."""
        return max(1, len(text) // 4)

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Splits document content into ordered chunks with section tracking."""
        raw_text = doc.content
        if not raw_text.strip():
            return []

        # 1. Identify sections by heading
        sections = self._split_by_headings(raw_text)
        chunks: List[Chunk] = []
        chunk_idx = 0
        global_char_offset = 0

        for section_heading, section_text in sections:
            section_chunks = self._chunk_section_text(section_text)
            for chunk_str in section_chunks:
                start = raw_text.find(chunk_str, global_char_offset)
                if start == -1:
                    start = global_char_offset
                end = start + len(chunk_str)
                global_char_offset = max(global_char_offset, start + len(chunk_str) - self.overlap)

                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_c{chunk_idx}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=chunk_str.strip(),
                    chunk_index=chunk_idx,
                    char_start=start,
                    char_end=end,
                    token_count_estimate=self.estimate_tokens(chunk_str),
                    section_heading=section_heading,
                    source_url=doc.metadata.get("source_url"),
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks

    def _split_by_headings(self, text: str) -> List[tuple[Optional[str], str]]:
        matches = list(self.heading_pattern.finditer(text))
        if not matches:
            return [(None, text)]

        sections = []
        # Pre-heading text
        if matches[0].start() > 0:
            pre_text = text[: matches[0].start()].strip()
            if pre_text:
                sections.append((None, pre_text))

        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((heading, f"{heading}\n\n{body}"))

        return sections

    def _chunk_section_text(self, text: str) -> List[str]:
        if len(text) <= self.target_size:
            return [text] if len(text) >= self.min_chunk_size else [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current = []
        current_len = 0

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue

            if len(p_clean) > self.target_size:
                # Split large paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", p_clean)
                for s in sentences:
                    if current_len + len(s) > self.target_size and current:
                        chunks.append(" ".join(current))
                        # Retain overlap from end of current
                        overlap_text = current[-1] if len(current[-1]) <= self.overlap else current[-1][-self.overlap :]
                        current = [overlap_text, s]
                        current_len = len(overlap_text) + len(s)
                    else:
                        current.append(s)
                        current_len += len(s)
            else:
                if current_len + len(p_clean) > self.target_size and current:
                    chunks.append("\n\n".join(current))
                    overlap_item = current[-1] if len(current[-1]) <= self.overlap else current[-1][-self.overlap :]
                    current = [overlap_item, p_clean]
                    current_len = len(overlap_item) + len(p_clean)
                else:
                    current.append(p_clean)
                    current_len += len(p_clean)

        if current:
            chunk_cand = "\n\n".join(current)
            if len(chunk_cand) >= self.min_chunk_size or not chunks:
                chunks.append(chunk_cand)

        return chunks
