"""Document and chunk metadata definitions."""

import hashlib
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    content: str
    source: str = "manual"
    category: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count_estimate: int
    section_heading: Optional[str] = None
    source_url: Optional[str] = None
    sha256_hash: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.sha256_hash:
            self.sha256_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
