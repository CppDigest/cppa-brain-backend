"""
Pinecone ingestion module for document indexing and vector storage.

Handles Pinecone index creation, document chunking, and vector upsert operations.
Uses Pinecone's integrated cloud embeddings for hybrid search (dense + sparse).

Note: Document retrieval/search functionality is handled by query.py
"""

import logging
import re
from typing import List, Dict, Any, Optional
import hashlib

try:
    from pinecone import Pinecone
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError as e:
    Pinecone = None  # type: ignore[assignment]
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

from config import PineconeConfig, EmbeddingConfig

logger = logging.getLogger(__name__)


class PineconeIngestion:
    """
    Handles Pinecone index creation, document chunking, and vector upsert operations.

    Provides functionality for:
    - Creating and managing Pinecone indexes (dense and sparse)
    - Chunking documents with configurable parameters
    - Filtering invalid chunks
    - Upserting documents to Pinecone with error tracking
    - Retrieving index statistics
    """

    def __init__(
        self,
    ):
        """
        Initialize PineconeIngestion with configuration from environment variables.

        Loads PineconeConfig and EmbeddingConfig automatically.
        Sets up client, text splitter, and index references (lazy initialization).
        """
        self._validate_imports()

        self.config = PineconeConfig()
        self.embedding_config = EmbeddingConfig()

        self._setup_client()
        self._initialize_text_splitter()
        self._setup_indexes()

        logger.info(
            f"Using Pinecone hybrid search with dense model: {self.embedding_config.pinecone_model} "
            f"and sparse model: {self.embedding_config.pinecone_sparse_model}"
        )

    def _validate_imports(self) -> None:
        """Validate that required imports are available."""
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Missing optional dependencies required for Pinecone ingestion. "
                "Install with: pip install pinecone-client langchain-text-splitters"
            ) from _IMPORT_ERROR

    def _setup_client(self) -> None:
        """Set up Pinecone client variables."""
        self.pc: Optional[Pinecone] = None
        self._pc_initialized = False

    def _initialize_text_splitter(self) -> None:
        """Initialize text splitter for document chunking."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )

    def _setup_indexes(self) -> None:
        """Set up index references."""
        self.dense_index: Optional[Any] = None
        self.sparse_index: Optional[Any] = None
        self._dense_index_initialized = False
        self._sparse_index_initialized = False

    def _ensure_pinecone_client(self) -> None:
        """Lazily initialize Pinecone client if not already initialized."""
        if not self._pc_initialized:
            try:
                self.pc = Pinecone(api_key=self.config.api_key)
                self._pc_initialized = True
                logger.info("Pinecone client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone client: {e}")
                raise ConnectionError(
                    f"Cannot connect to Pinecone. Check your internet connection and API key. "
                    f"Error: {e}"
                ) from e

    def _get_or_create_indexes(self) -> None:
        """
        Get existing indexes or create new ones if they don't exist.
        Uses lazy initialization to avoid connection errors during __init__.
        Creates both dense and sparse indexes for hybrid search.
        """
        if self._dense_index_initialized and self._sparse_index_initialized:
            return

        try:
            self._ensure_pinecone_client()
            if self.pc is None:
                raise RuntimeError("Pinecone client not initialized")

            existing_indexes = {idx.name for idx in self.pc.list_indexes()}
            dense_name = self.config.index_name
            sparse_name = f"{self.config.index_name}-sparse"

            if self._indexes_exist(existing_indexes, dense_name, sparse_name):
                self._connect_to_existing_indexes(dense_name, sparse_name)
            else:
                self._create_new_indexes(existing_indexes, dense_name, sparse_name)

            self._dense_index_initialized = True
            self._sparse_index_initialized = True

        except ConnectionError as e:
            logger.error(f"Network error connecting to Pinecone: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating/getting Pinecone indexes: {e}")
            raise

    def _indexes_exist(
        self, existing_indexes: set, dense_name: str, sparse_name: str
    ) -> bool:
        """Check if both dense and sparse indexes exist."""
        return dense_name in existing_indexes and sparse_name in existing_indexes

    def _connect_to_existing_indexes(self, dense_name: str, sparse_name: str) -> None:
        """Connect to existing Pinecone indexes."""
        logger.info(f"Using existing indexes: {dense_name} and {sparse_name}")
        if self.pc is None:
            raise RuntimeError("Pinecone client not initialized")
        self.dense_index = self.pc.Index(dense_name)
        self.sparse_index = self.pc.Index(sparse_name)

    def _create_new_indexes(
        self, existing_indexes: set, dense_name: str, sparse_name: str
    ) -> None:
        """Create new dense and sparse indexes if they don't exist."""
        logger.info(
            f"Creating new hybrid indexes in region {self.config.environment}: "
            f"{dense_name} (dense) and {sparse_name} (sparse)"
        )
        try:
            if self.pc is None:
                raise RuntimeError("Pinecone client not initialized")

            if dense_name not in existing_indexes:
                self._create_pinecone_index(
                    dense_name, self.embedding_config.pinecone_model
                )
            if sparse_name not in existing_indexes:
                self._create_pinecone_index(
                    sparse_name, self.embedding_config.pinecone_sparse_model
                )

            self.dense_index = self.pc.Index(dense_name)
            self.sparse_index = self.pc.Index(sparse_name)
        except Exception as create_error:
            self._handle_index_creation_error(create_error)

    def _create_pinecone_index(self, index_name: str, model_name: str) -> None:
        """Create Pinecone index with integrated embedding model."""
        logger.info(f"Creating index '{index_name}' with model: {model_name}")
        if self.pc is None:
            raise RuntimeError("Pinecone client not initialized")
        self.pc.create_index_for_model(
            name=index_name,
            cloud=self.config.cloud,
            region=self.config.environment,
            embed={
                "model": model_name,
                "field_map": {"text": "chunk_text"},
            },
        )

    def _handle_index_creation_error(self, error: Exception) -> None:
        """Handle errors during index creation."""
        error_msg = str(error)
        if "NOT_FOUND" in error_msg or "not found" in error_msg.lower():
            raise ValueError(
                f"Invalid Pinecone region: '{self.config.environment}'. "
                f"Valid regions include: us-east-1, us-west-2, eu-west-1, ap-southeast-1, etc. "
                f"Check your PINECONE_ENVIRONMENT setting. Error: {error}"
            ) from error
        raise error

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Document]:
        """
        Split documents into chunks using config defaults or provided parameters.

        Args:
            documents: List of documents to chunk
            chunk_size: Optional chunk size (uses config default if not provided)
            chunk_overlap: Optional chunk overlap (uses config default if not provided)

        Returns:
            List of chunked and filtered Document objects
        """
        try:
            # Use provided parameters or fall back to config defaults
            size = chunk_size if chunk_size is not None else self.config.chunk_size
            overlap = (
                chunk_overlap
                if chunk_overlap is not None
                else self.config.chunk_overlap
            )

            # Update text splitter if parameters differ from current config
            if size != self.config.chunk_size or overlap != self.config.chunk_overlap:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=size,
                    chunk_overlap=overlap,
                    length_function=len,
                )

            chunks = self.text_splitter.split_documents(documents)
            # Filter out chunks with empty, very short, or meaningless text
            valid_chunks = self._filter_valid_chunks(chunks)
            filtered_count = len(chunks) - len(valid_chunks)
            if filtered_count > 0:
                logger.warning(
                    f"Filtered out {filtered_count} chunks with empty, too short, or meaningless text"
                )
            logger.info(
                f"Split {len(documents)} documents into {len(valid_chunks)} valid chunks"
            )
            return valid_chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise

    def _filter_valid_chunks(self, chunks: List[Document]) -> List[Document]:
        """Filter out chunks with empty, very short, or meaningless text."""

        MIN_TEXT_LENGTH = 10  # Minimum characters required for sparse vector generation
        MIN_WORDS = 3  # Minimum number of actual words

        valid_chunks = []
        for chunk in chunks:
            text = chunk.page_content.strip() if chunk.page_content else ""

            # Skip empty or too short text
            if not text or len(text) < MIN_TEXT_LENGTH:
                continue

            # Filter out markdown table separators and similar patterns
            # Patterns like: "| --- | --- |", "|---|", "| - | - |", etc.
            table_separator_pattern = r"^\|[\s\-:]+\|[\s\-:]*\|?[\s\-:]*\|?.*$"
            if re.match(table_separator_pattern, text):
                continue

            # Filter out text that's mostly formatting characters (pipes, dashes, spaces)
            # If more than 70% of characters are formatting, skip it
            formatting_chars = len(re.findall(r"[|\-\s:]", text))
            if len(text) > 0 and formatting_chars / len(text) > 0.7:
                continue

            # Filter out text with too few actual words
            words = re.findall(r"\b[a-zA-Z0-9]+\b", text)
            if len(words) < MIN_WORDS:
                continue

            # If more than 50% of non-space characters are punctuation, skip it
            non_space_chars = re.findall(r"[^\s]", text)
            punctuation_chars = len(re.findall(r"[^\w\s]", text))
            if (
                len(non_space_chars) > 0
                and punctuation_chars / len(non_space_chars) > 0.5
            ):
                continue

            valid_chunks.append(chunk)
        return valid_chunks

    def upsert_documents(
        self,
        documents: List[Document],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upsert documents to Pinecone indexes using cloud embeddings.

        Documents are automatically chunked, filtered, and upserted to both
        dense and sparse indexes. Returns statistics including failed documents.

        Args:
            documents: List of Document objects to upsert
            namespace: Optional Pinecone namespace (default: None)

        Returns:
            Dictionary with keys:
                - upserted: Number of successfully upserted documents
                - total: Total number of input documents
                - errors: List of error messages
                - failed_documents: List of failed document info dicts
                - failed_count: Count of failed documents
        """
        try:
            if not documents:
                logger.warning("No documents to upsert")
                return {"upserted": 0, "errors": [], "failed_documents": []}

            self._ensure_indexes_ready()
            chunked_documents = self.chunk_documents(documents)
            total_upserted, errors, failed_documents = self._upsert_all_batches(
                chunked_documents, namespace
            )
            return self._build_upsert_result(
                total_upserted, len(documents), errors, failed_documents
            )

        except Exception as e:
            logger.error(f"Error in upsert_documents: {e}")
            raise

    def _mark_batch_failed(
        self, batch: List[Document], error: Exception, start_idx: int
    ) -> List[Dict[str, Any]]:
        """Mark all documents in a failed batch as failed."""
        failed = []
        for idx, doc in enumerate(batch):
            meta = doc.metadata or {}
            failed.append(
                {
                    "doc_id": meta.get("doc_id", f"doc_{start_idx}_{idx}"),
                    "type": meta.get("type", "unknown"),
                    "reason": f"Batch upsert failed: {str(error)}",
                    "text_length": len(doc.page_content) if doc.page_content else 0,
                    "metadata": meta,
                }
            )
        return failed

    def _upsert_all_batches(
        self,
        documents: List[Document],
        namespace: Optional[str],
    ) -> tuple[int, List[str], List[Dict[str, Any]]]:
        """Upsert all document batches and return statistics with failed documents."""
        total_upserted, errors, failed_docs = 0, [], []
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            try:
                records, filtered = self._prepare_batch_records(batch, i)
                if filtered:
                    failed_docs.extend(filtered)
                    logger.warning(
                        f"Batch {batch_num}: {len(filtered)} documents filtered"
                    )

                if not records:
                    logger.warning(
                        f"Batch {batch_num}: no valid records ({len(filtered)} filtered)"
                    )
                    continue

                self._upsert_batch(records, namespace, batch_num)
                total_upserted += len(records)
                logger.info(
                    f"Upserted batch {batch_num}: {len(records)}/{len(batch)} documents"
                )
            except Exception as e:
                error_msg = f"Error upserting batch {batch_num}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                failed_docs.extend(self._mark_batch_failed(batch, e, i))

        return total_upserted, errors, failed_docs

    def _build_upsert_result(
        self,
        total_upserted: int,
        total: int,
        errors: List[str],
        failed_documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build and log upsert result dictionary."""
        result = {
            "upserted": total_upserted,
            "total": total,
            "errors": errors,
            "failed_documents": failed_documents,
            "failed_count": len(failed_documents),
        }
        logger.info(
            f"Upsert complete: {result['upserted']}/{result['total']} documents, "
            f"{result['failed_count']} failed"
        )
        if failed_documents:
            logger.warning(
                f"Failed documents summary: {result['failed_count']} documents failed to upsert"
            )
        return result

    def _ensure_indexes_ready(self) -> None:
        """Ensure indexes are initialized and ready for use."""
        if not self._dense_index_initialized or not self._sparse_index_initialized:
            self._get_or_create_indexes()
        if self.dense_index is None or self.sparse_index is None:
            raise RuntimeError("Pinecone indexes not initialized")

    def _prepare_batch_records(
        self, batch: List[Document], batch_start_idx: int
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare records for Pinecone upsert from a batch of documents."""
        MIN_TEXT_LENGTH = 10  # Minimum characters required for sparse vector generation
        records = []
        failed_documents = []
        for doc in batch:
            text = doc.page_content.strip() if doc.page_content else ""
            metadata = doc.metadata or {}
            original_doc_id = (
                metadata.get("doc_id") or f"doc_{batch_start_idx}_{len(records)}"
            )

            # Skip documents with empty or too short text
            if not text or len(text) < MIN_TEXT_LENGTH:
                failed_documents.append(
                    {
                        "doc_id": original_doc_id,
                        "type": metadata.get("type", "unknown"),
                        "reason": f"Text too short (length: {len(text)}, minimum: {MIN_TEXT_LENGTH})",
                        "text_length": len(text),
                        "metadata": metadata,
                    }
                )
                continue

            doc_id = f"{original_doc_id}_{text[:50]}_{len(text)}"
            doc_id = hashlib.md5(doc_id.encode()).hexdigest()

            record = {
                "id": doc_id,
                "chunk_text": text,
            }
            if metadata:
                record.update(metadata)
            records.append(record)
        return records, failed_documents

    def _upsert_batch(
        self,
        records: List[Dict[str, Any]],
        namespace: Optional[str],
        batch_num: int,
    ) -> None:
        """Upsert a batch of records to both dense and sparse indexes."""
        if self.dense_index is None or self.sparse_index is None:
            raise RuntimeError("Pinecone indexes not initialized")
        try:
            self.dense_index.upsert_records(records=records, namespace=namespace)
        except Exception as e:
            logger.error(
                f"Failed to upsert batch {batch_num} to dense index: {e}. "
                f"Records: {[r.get('id', 'unknown') for r in records]}"
            )
            raise
        try:
            self.sparse_index.upsert_records(records=records, namespace=namespace)
        except Exception as e:
            logger.error(
                f"Failed to upsert batch {batch_num} to sparse index: {e}. "
                f"Records: {[r.get('id', 'unknown') for r in records]}"
            )
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone indexes.

        Returns:
            Dictionary with index statistics for both dense and sparse indexes
        """
        try:
            self._ensure_indexes_ready()
            dense_stats = self.dense_index.describe_index_stats()  # type: ignore
            sparse_stats = self.sparse_index.describe_index_stats()  # type: ignore
            return self._format_index_stats(dense_stats, sparse_stats)
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return self._get_empty_stats(str(e))

    def _format_index_stats(
        self, dense_stats: Dict[str, Any], sparse_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format index statistics into structured dictionary."""
        return {
            "dense_index": {
                "total_vectors": dense_stats.get("total_vector_count", 0),
                "dimension": dense_stats.get("dimension", 0),
                "index_fullness": dense_stats.get("index_fullness", 0),
                "namespaces": dense_stats.get("namespaces", {}),
            },
            "sparse_index": {
                "total_vectors": sparse_stats.get("total_vector_count", 0),
                "dimension": sparse_stats.get("dimension", 0),
                "index_fullness": sparse_stats.get("index_fullness", 0),
                "namespaces": sparse_stats.get("namespaces", {}),
            },
        }

    def _get_empty_stats(self, error_msg: str) -> Dict[str, Any]:
        """Return empty stats structure with error message."""
        empty_stats = {
            "total_vectors": 0,
            "dimension": 0,
            "index_fullness": 0,
            "namespaces": {},
        }
        return {
            "error": error_msg,
            "dense_index": empty_stats.copy(),
            "sparse_index": empty_stats.copy(),
        }
