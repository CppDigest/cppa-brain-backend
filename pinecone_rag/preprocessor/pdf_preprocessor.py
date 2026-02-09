"""
PDF preprocessing for blog/post PDFs (e.g. Bjarne Stroustrup papers).

Extracts text per page and produces LangChain Documents for Pinecone RAG.
Uses PyMuPDF (fitz) for higher-quality text extraction and layout handling.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from config import BlogPdfConfig
from preprocessor.utility import (
    validate_content_length,
    clean_text,
    get_timestamp_from_date,
    timestamp_from_pdf_date,
)

logger = logging.getLogger(__name__)


def _get_fitz():
    """Lazy import PyMuPDF to avoid import error when package not installed."""
    try:
        import fitz

        return fitz
    except ImportError:
        raise ImportError(
            "PDF processing requires PyMuPDF. Install with: pip install pymupdf"
        ) from None


class PdfPreprocessor:
    """
    Process PDF files from a directory (e.g. Bjarne Stroustrup blog) for RAG.

    Uses PyMuPDF (fitz) for higher-quality text extraction. Emits one Document
    per page; metadata includes page number, title from PDF metadata, and safe creation date.
    """

    def __init__(self, config: Optional[BlogPdfConfig] = None):
        self.config = config or BlogPdfConfig()
        self.data_dir = Path(self.config.data_dir)
        self.author = self.config.author
        self.source_url = self.config.source_url

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        """
        Load all PDFs from the configured directory and convert to Documents.

        Each page becomes one Document (page_content = page text, metadata includes
        doc_id, type, author, title, link, page, total_pages, created_at). Title
        from PDF /Title metadata or filename stem; creation date from metadata when valid.
        """
        if not self.data_dir.exists():
            logger.warning("PDF data dir does not exist: %s", self.data_dir)
            return []

        fitz = _get_fitz()
        documents: List[Document] = []
        pdf_paths = sorted(self.data_dir.glob("*.pdf"))

        if limit is not None:
            pdf_paths = pdf_paths[:limit]

        for pdf_path in pdf_paths:
            try:
                docs = self._pdf_to_documents(pdf_path, fitz)
                documents.extend(docs)
            except OSError as e:
                logger.exception("Error reading PDF %s: %s", pdf_path.name, e)
            except Exception as e:  # pylint: disable=broad-except
                # Skip malformed or unsupported PDFs without failing the whole batch
                logger.exception("Error processing PDF %s: %s", pdf_path.name, e)

        logger.info(
            "Loaded %d page documents from %d PDFs in %s",
            len(documents),
            len(pdf_paths),
            self.data_dir,
        )
        return documents

    def _pdf_doc_metadata(self, doc, pdf_path: Path) -> tuple:
        """Extract title, date, timestamp, URL, and page count. Returns (title, created_at, time_stamp, url, total_pages)."""
        total_pages = len(doc)
        meta = doc.metadata or {}
        stem = pdf_path.stem
        parts = stem.split("&&")
        title = parts[-1] if len(parts) > 1 else stem
        raw_creation = meta.get("creationDate") or meta.get("/CreationDate")
        created_at = timestamp_from_pdf_date(raw_creation) if raw_creation else None
        if created_at is None:
            created_at = datetime.now().date().isoformat()
        time_stamp = timestamp_from_pdf_date(raw_creation)
        url = (
            f"{self.source_url}/{parts[0]}.pdf"
            if parts
            else f"{self.source_url}/{pdf_path.name}"
        )
        return title, created_at, time_stamp, url, total_pages

    def _pdf_to_documents(self, pdf_path: Path, fitz) -> List[Document]:
        """Convert one PDF file to a list of Documents (one per page) using PyMuPDF."""
        doc = fitz.open(str(pdf_path))
        title, _created_at, time_stamp, url, total_pages = self._pdf_doc_metadata(
            doc, pdf_path
        )
        documents: List[Document] = []
        for page_num in range(total_pages):
            text = doc[page_num].get_text()
            if not validate_content_length(text, min_length=30):
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": f"{url}#p{page_num + 1}",
                        "type": "blog-pdf",
                        "author": self.author,
                        "title": title,
                        "url": url,
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "timestamp": time_stamp,
                    },
                )
            )
        return documents
