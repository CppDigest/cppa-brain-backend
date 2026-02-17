"""
Pinecone ingestion module for document indexing and vector storage.

Handles Pinecone index creation, document chunking, and vector operations (upsert, update, delete).
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
    """Handles Pinecone index creation, document chunking, and vector operations."""

    def __init__(self):
        """Initialize with configuration from environment variables."""
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
        """Validate required imports."""
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Missing optional dependencies required for Pinecone ingestion. "
                "Install with: pip install pinecone-client langchain-text-splitters"
            ) from _IMPORT_ERROR

    def _setup_client(self) -> None:
        """Set up Pinecone client."""
        self.pc: Optional[Pinecone] = None
        self._pc_initialized = False

    def _initialize_text_splitter(self) -> None:
        """Initialize text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def _setup_indexes(self) -> None:
        """Set up index references."""
        self.dense_index: Optional[Any] = None
        self.sparse_index: Optional[Any] = None
        self._dense_index_initialized = False
        self._sparse_index_initialized = False

    def _ensure_pinecone_client(self) -> None:
        """Initialize Pinecone client if needed."""
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
        """Get existing indexes or create new ones if they don't exist."""
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
        """Check if indexes exist."""
        return dense_name in existing_indexes and sparse_name in existing_indexes

    def _connect_to_existing_indexes(self, dense_name: str, sparse_name: str) -> None:
        """Connect to existing indexes."""
        logger.info(f"Using existing indexes: {dense_name} and {sparse_name}")
        if self.pc is None:
            raise RuntimeError("Pinecone client not initialized")
        self.dense_index = self.pc.Index(dense_name)
        self.sparse_index = self.pc.Index(sparse_name)

    def _create_new_indexes(
        self, existing_indexes: set, dense_name: str, sparse_name: str
    ) -> None:
        """Create new indexes if they don't exist."""
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
        """Create Pinecone index with embedding model."""
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
        """Handle index creation errors."""
        error_msg = str(error)
        if "NOT_FOUND" in error_msg or "not found" in error_msg.lower():
            raise ValueError(
                f"Invalid Pinecone region: '{self.config.environment}'. "
                f"Valid regions include: us-east-1, us-west-2, eu-west-1, ap-southeast-1, etc. "
                f"Check your PINECONE_ENVIRONMENT setting. Error: {error}"
            ) from error
        raise error

    def _is_valid_chunk(self, text: str, min_length: int, min_words: int) -> bool:
        """Check if chunk is valid."""
        if not text or len(text) < min_length:
            return False
        if self._is_table_separator(text):
            return False
        if self._is_mostly_formatting(text):
            return False
        if self._has_too_few_words(text, min_words):
            return False
        if self._is_mostly_punctuation(text):
            return False
        return True

    def _is_table_separator(self, text: str) -> bool:
        """Check if text is table separator."""
        pattern = r"^\|[\s\-:]+\|[\s\-:]*\|?[\s\-:]*\|?.*$"
        return bool(re.match(pattern, text))

    def _is_mostly_formatting(self, text: str) -> bool:
        """Check if text is mostly formatting."""
        formatting_chars = len(re.findall(r"[|\-\s:]", text))
        return len(text) > 0 and formatting_chars / len(text) > 0.7

    def _has_too_few_words(self, text: str, min_words: int) -> bool:
        """Check if text has too few words."""
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text)
        return len(words) < min_words

    def _is_mostly_punctuation(self, text: str) -> bool:
        """Check if text is mostly punctuation."""
        non_space_chars = re.findall(r"[^\s]", text)
        punctuation_chars = len(re.findall(r"[^\w\s]", text))
        return (
            len(non_space_chars) > 0 and punctuation_chars / len(non_space_chars) > 0.5
        )

    def upsert_documents(
        self,
        documents: List[Document],
        namespace: Optional[str] = None,
        is_chunked: bool = False,
    ) -> Dict[str, Any]:
        """Upsert documents to Pinecone indexes. Returns statistics."""
        try:
            if not documents:
                logger.warning("No documents to upsert")
                return {"upserted": 0, "errors": [], "failed_documents": []}

            self._ensure_indexes_ready()
            if not is_chunked:
                chunked_documents = self.text_splitter.split_documents(documents)
            else:
                chunked_documents = documents
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
        """Mark batch documents as failed."""
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
        """Upsert all batches."""
        total_upserted, errors, failed_docs = 0, [], []
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            try:
                records = self._prepare_batch_records(batch, i)

                if not records:
                    logger.warning(f"Batch {batch_num}: no valid records")
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
        """Build upsert result."""
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
        """Ensure indexes are ready."""
        if not self._dense_index_initialized or not self._sparse_index_initialized:
            self._get_or_create_indexes()
        if self.dense_index is None or self.sparse_index is None:
            raise RuntimeError("Pinecone indexes not initialized")

    def _prepare_batch_records(
        self, batch: List[Document], batch_start_idx: int
    ) -> List[Dict[str, Any]]:
        """Prepare batch records for upsert."""
        records = []

        for doc in batch:
            text = doc.page_content.strip() if doc.page_content else ""
            if not self._is_valid_chunk(
                text, self.config.min_text_length, self.config.min_words
            ):
                continue

            metadata = doc.metadata or {}
            if "title" in metadata and metadata["title"] != "":
                text = f"Title: {metadata['title']}\n\n{text}"

            original_doc_id = metadata.get(
                "doc_id", metadata.get("url", f"doc_{batch_start_idx}_{len(records)}")
            )
            if "start_index" in metadata:
                original_doc_id = f"{original_doc_id}_{metadata['start_index']}"
            else:
                original_doc_id = f"{original_doc_id}_{text[:50]}_{len(text)}"

            doc_id = hashlib.md5(original_doc_id.encode()).hexdigest()

            record = {"id": doc_id, "chunk_text": text}
            if metadata:
                record.update(metadata)
            records.append(record)
        return records

    def _create_failed_doc_info(
        self, doc_id: str, metadata: Dict[str, Any], text: str, min_length: int
    ) -> Dict[str, Any]:
        """Create failed document info."""
        return {
            "doc_id": doc_id,
            "type": metadata.get("type", "unknown"),
            "reason": f"Text too short (length: {len(text)}, minimum: {min_length})",
            "text_length": len(text),
            "metadata": metadata,
        }

    def _generate_chunk_id(self, original_doc_id: str, text: str) -> str:
        """Generate chunk ID."""
        doc_id = f"{original_doc_id}_{text[:50]}_{len(text)}"
        return hashlib.md5(doc_id.encode()).hexdigest()

    def _upsert_batch(
        self,
        records: List[Dict[str, Any]],
        namespace: Optional[str],
        batch_num: int,
    ) -> None:
        """Upsert batch to both indexes."""
        self._ensure_indexes_ready()
        self._upsert_to_index(self.dense_index, records, namespace, batch_num, "dense")
        self._upsert_to_index(
            self.sparse_index, records, namespace, batch_num, "sparse"
        )

    def _upsert_to_index(
        self,
        index: Any,
        records: List[Dict[str, Any]],
        namespace: Optional[str],
        batch_num: int,
        index_type: str,
    ) -> None:
        """Upsert to single index."""
        try:
            index.upsert_records(records=records, namespace=namespace)
        except Exception as e:
            record_ids = [r.get("id", "unknown") for r in records]
            logger.error(
                f"Failed to upsert batch {batch_num} to {index_type} index: {e}. "
                f"Records: {record_ids}"
            )
            raise

    def update_documents(
        self,
        documents: List[Document],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update existing documents in Pinecone indexes."""
        try:
            if not documents:
                logger.warning("No documents to update")
                return {"updated": 0, "errors": []}

            self._ensure_indexes_ready()
            chunked_documents = self.chunk_documents(documents)
            updated, errors = self._update_all_batches(chunked_documents, namespace)

            result = {
                "updated": updated,
                "total": len(documents),
                "errors": errors,
            }
            logger.info(
                f"Update complete: {result['updated']}/{result['total']} documents"
            )
            return result
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise

    def _update_all_batches(
        self,
        documents: List[Document],
        namespace: Optional[str],
    ) -> tuple[int, List[str]]:
        """Update all batches."""
        total_updated, errors = 0, []
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            try:
                records, _ = self._prepare_batch_records(batch, i)
                if not records:
                    continue
                self._update_batch(records, namespace, batch_num)
                total_updated += len(records)
            except Exception as e:
                error_msg = f"Error updating batch {batch_num}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return total_updated, errors

    def _update_batch(
        self,
        records: List[Dict[str, Any]],
        namespace: Optional[str],
        batch_num: int,
    ) -> None:
        """Update batch in both indexes."""
        self._ensure_indexes_ready()
        self._upsert_to_index(self.dense_index, records, namespace, batch_num, "dense")
        self._upsert_to_index(
            self.sparse_index, records, namespace, batch_num, "sparse"
        )
        logger.info(f"Updated batch {batch_num}: {len(records)} documents")

    def delete_documents(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete documents from Pinecone indexes by IDs."""
        try:
            if not ids:
                logger.warning("No document IDs to delete")
                return {"deleted": 0, "errors": []}

            self._ensure_indexes_ready()
            deleted, errors = self._delete_all_batches(ids, namespace)

            result = {
                "deleted": deleted,
                "total": len(ids),
                "errors": errors,
            }
            logger.info(
                f"Delete complete: {result['deleted']}/{result['total']} documents"
            )
            return result
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def _delete_all_batches(
        self,
        ids: List[str],
        namespace: Optional[str],
    ) -> tuple[int, List[str]]:
        """Delete all batches."""
        total_deleted, errors = 0, []
        batch_size = self.config.batch_size

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            try:
                self._delete_batch(batch_ids, namespace, batch_num)
                total_deleted += len(batch_ids)
            except Exception as e:
                error_msg = f"Error deleting batch {batch_num}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return total_deleted, errors

    def _delete_batch(
        self,
        ids: List[str],
        namespace: Optional[str],
        batch_num: int,
    ) -> None:
        """Delete batch from both indexes."""
        self._ensure_indexes_ready()
        self._delete_from_index(self.dense_index, ids, namespace, batch_num, "dense")
        self._delete_from_index(self.sparse_index, ids, namespace, batch_num, "sparse")
        logger.info(f"Deleted batch {batch_num}: {len(ids)} documents")

    def _delete_from_index(
        self,
        index: Any,
        ids: List[str],
        namespace: Optional[str],
        batch_num: int,
        index_type: str,
    ) -> None:
        """Delete from single index."""
        try:
            index.delete(ids=ids, namespace=namespace)
        except Exception as e:
            logger.error(
                f"Failed to delete batch {batch_num} from {index_type} index: {e}"
            )
            raise

    def delete_namespace(self, namespace: str) -> None:
        """Delete namespace from Pinecone indexes."""
        self._ensure_indexes_ready()
        try:
            self.dense_index.delete_namespace(namespace=namespace)
            self.sparse_index.delete_namespace(namespace=namespace)
        except Exception as e:
            logger.error(f"Error deleting namespace: {e}")
            raise
        logger.info(f"Deleted namespace: {namespace}")
        return {
            "deleted": 1,
            "total": 1,
            "errors": [],
        }

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone indexes."""
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
        """Format index statistics."""
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
        """Return empty stats with error."""
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
