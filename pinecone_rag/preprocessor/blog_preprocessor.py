"""
Blog preprocessor: JSON files and PDFs under data/blog-posts.

Processes all JSON files (blog-html docs) and PDF files (via PdfPreprocessor)
under data/blog-posts to produce LangChain Documents with consistent metadata:
doc_id, title, url, author, timestamp, type, etc.

- JSON: doc_id = url (link), type = "blog-html", timestamp from published_parsed
  or parsed from filename if published_parsed is null.
- PDF: uses PdfPreprocessor per author subdir; metadata from pdf_preprocessor.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config import BlogConfig, BlogPdfConfig
from preprocessor.pdf_preprocessor import PdfPreprocessor
from preprocessor.utility import (
    validate_content_length,
    timestamp_from_published,
    timestamp_from_filename,
)

logger = logging.getLogger(__name__)


def _json_to_document(json_path: Path) -> Optional[Document]:
    """
    Build one Document from a blog JSON record.
    doc_id = url (link); type = blog-html; timestamp from published_parsed or filename.
    """
    raw = json_path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)
    url = data.get("link") or data.get("url") or ""
    if not url:
        logger.debug("Skip %s: no link/url", json_path.name)
        return None

    title = (data.get("title") or "").strip()
    author = (data.get("author") or "").strip()
    published_parsed = data.get("published_parsed")
    timestamp = None
    if published_parsed is None or published_parsed == "":
        timestamp = timestamp_from_filename(json_path.name)
    else:
        timestamp = timestamp_from_published(published_parsed)

    content = (data.get("content") or data.get("summary") or "").strip()
    if not validate_content_length(content, min_length=50):
        logger.debug("Skip %s: content too short", json_path.name)
        return None

    meta = {
        "title": title,
        "url": url,
        "author": author,
        "timestamp": timestamp,
        "type": "blog-html",
    }
    return Document(page_content=content, metadata=meta)


class BlogPreprocessor:
    """
    Process all blog JSON files and PDFs under data/blog-posts.

    JSON files become one Document each (type blog-html). PDFs are processed
    via PdfPreprocessor per author subdir (type blog-pdf, metadata from pdf_preprocessor).
    """

    def __init__(self, config: Optional[BlogConfig] = None):
        self.config = config or BlogConfig()
        self.data_dir = Path(self.config.data_dir)
        self.include_pdf = self.config.include_pdf

    def _load_json_documents(
        self, limit: Optional[int] = None
    ) -> Tuple[List[Document], int]:
        """Load all JSON files under data_dir. Returns (documents, count)."""
        json_paths = sorted(self.data_dir.rglob("*.json"))
        if limit is not None:
            json_paths = json_paths[:limit]
        documents: List[Document] = []
        for json_path in json_paths:
            try:
                doc = _json_to_document(json_path)
                if doc is not None:
                    documents.append(doc)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Skip %s: %s", json_path.name, e)
        return documents, len(documents)

    def _load_pdf_documents(self, documents: List[Document]) -> None:
        """Extend documents with PDF page docs from each author subdir (in place)."""
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if not list(subdir.glob("*.pdf")):
                continue
            pdf_config = BlogPdfConfig(
                data_dir=str(subdir),
                author=subdir.name,
                namespace=self.config.namespace,
                source_url=_default_source_url_for_author(subdir.name),
            )
            try:
                documents.extend(PdfPreprocessor(config=pdf_config).load_documents())
            except (OSError, ImportError) as e:
                logger.warning("Skip PDFs in %s: %s", subdir.name, e)
            except Exception as e:  # pylint: disable=broad-except
                # Skip malformed or unsupported PDF dir without failing the batch
                logger.warning("Skip PDFs in %s: %s", subdir.name, e)

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        """
        Load JSON and (optionally) PDF documents from data/blog-posts.

        Returns a single list: JSON docs (blog-html) first, then PDF page docs (blog-pdf).
        """
        if not self.data_dir.exists():
            logger.warning("Blog data dir does not exist: %s", self.data_dir)
            return []

        documents, json_count = self._load_json_documents(limit=limit)
        logger.info("Loaded %d blog-html documents from JSON", json_count)

        if self.include_pdf:
            self._load_pdf_documents(documents)

        logger.info(
            "Blog preprocessor total: %d documents (%d JSON + %d PDF pages)",
            len(documents),
            json_count,
            len(documents) - json_count,
        )
        return documents


def _default_source_url_for_author(author: str) -> str:
    """Default base URL for an author's PDFs (stroustrup.com, etc.)."""
    lower = author.lower()
    if "stroustrup" in lower:
        return "https://stroustrup.com"
    if "sutter" in lower:
        return "https://herbsutter.com"
    if "boccara" in lower:
        return "https://www.fluentcpp.com"
    if "grimm" in lower:
        return "https://www.modernescpp.com"
    if "niebler" in lower:
        return "https://ericniebler.com"
    return "https://example.com"
