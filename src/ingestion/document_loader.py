"""Document loader for local markdown, text files, and JSON payloads."""

import glob
import os
from typing import Dict, List, Optional
from src.ingestion.metadata import Document


class DocumentLoader:
    """Loads documents from raw dictionaries, markdown files, or directories."""

    @classmethod
    def from_text(cls, doc_id: str, title: str, text: str, source: str = "direct", metadata: Optional[Dict] = None) -> Document:
        return Document(
            doc_id=doc_id,
            title=title,
            content=text,
            source=source,
            metadata=metadata or {},
        )

    @classmethod
    def from_file(cls, filepath: str) -> Document:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Document file not found: {filepath}")

        filename = os.path.basename(filepath)
        doc_id = os.path.splitext(filename)[0]

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        title = doc_id.replace("_", " ").title()
        # If first line is a markdown title, extract it
        lines = content.splitlines()
        if lines and lines[0].startswith("# "):
            title = lines[0].lstrip("# ").strip()

        return Document(
            doc_id=doc_id,
            title=title,
            content=content,
            source=filepath,
            metadata={"filepath": filepath, "extension": os.path.splitext(filename)[1]},
        )

    @classmethod
    def from_directory(cls, dirpath: str, glob_pattern: str = "*.md") -> List[Document]:
        if not os.path.exists(dirpath):
            return []
        pattern = os.path.join(dirpath, glob_pattern)
        files = glob.glob(pattern)
        return [cls.from_file(f) for f in files]
