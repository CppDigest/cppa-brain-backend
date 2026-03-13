"""
Pinecone ingestion module for document indexing and vector storage.

Handles Pinecone index creation, document chunking, and vector operations (upsert, update, delete).
Uses Pinecone's integrated cloud embeddings for hybrid search (dense + sparse).

Note: Document retrieval/search functionality is handled by query.py
"""

import logging
import re
import requests
from typing import List, Dict, Any, Optional
import hashlib
from enum import Enum
from tqdm import tqdm

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

from config import PineconeConfig, EmbeddingConfig, PineconeInstance

logger = logging.getLogger(__name__)


class ProcessType(str, Enum):
    """Supported processing operations for Pinecone document handling."""

    UPSERT = "upsert"
    UPDATE_METADATA_BY_ID = "update_metadata_by_id"
    UPDATE_DOCUMENT_BY_ID = "update_document_by_id"
    UPDATE_VALUE_BY_ID = "update_value_by_id"

    @classmethod
    def from_value(cls, value: "ProcessType | str") -> "ProcessType":
        """Parse process type from enum or string value."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(v.value for v in cls)
            raise ValueError(
                f"Invalid process_type: {value}. Use one of: {allowed}"
            ) from exc


class PineconeIngestion:
    """Handles Pinecone index creation, document chunking, and vector operations."""

    def __init__(self, instance: PineconeInstance = PineconeInstance.PUBLIC):
        """Initialize with configuration from environment variables.

        Args:
            instance: Whether to use the public or private Pinecone API key.
        """
        self._validate_imports()

        self.config = PineconeConfig()
        self.embedding_config = EmbeddingConfig()
        self.instance = instance

        self._setup_client()
        self._initialize_text_splitter()
        self._setup_indexes()

        logger.info(
            f"Using Pinecone hybrid search with dense model: {self.embedding_config.pinecone_model} "
            f"and sparse model: {self.embedding_config.pinecone_sparse_model} "
            f"(instance: {self.instance.value})"
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
                self.pc = Pinecone(api_key=self._active_api_key)
                self._pc_initialized = True
                logger.info(
                    "Pinecone client initialized (instance: %s)", self.instance.value
                )
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone client: {e}")
                raise ConnectionError(
                    f"Cannot connect to Pinecone. Check your internet connection and API key. "
                    f"Error: {e}"
                ) from e

    @property
    def _active_api_key(self) -> str:
        """Return the API key for the currently selected instance."""
        if self.instance == PineconeInstance.PRIVATE:
            key = self.config.private_api_key
            if not (key and key.strip()):
                raise ValueError(
                    "PineconeInstance.PRIVATE is selected but PINECONE_PRIVATE_API_KEY "
                    "is not set or is empty. Set the environment variable before using the private instance."
                )
            return key
        return self.config.api_key

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
                self._create_sparse_pinecone_index(
                    sparse_name, self.embedding_config.pinecone_sparse_model
                )

            self.dense_index = self.pc.Index(dense_name)
            self.sparse_index = self.pc.Index(sparse_name)
        except Exception as create_error:
            self._handle_index_creation_error(create_error)

    def _create_pinecone_index(self, index_name: str, model_name: str) -> None:
        """Create Pinecone index with embedding model and write/read parameters."""
        logger.info(f"Creating index '{index_name}' with model: {model_name}")
        if self.pc is None:
            raise RuntimeError("Pinecone client not initialized")
        write_params = {
            "input_type": self.embedding_config.embed_write_input_type,
            "truncate": self.embedding_config.embed_truncate,
        }
        read_params = {
            "input_type": self.embedding_config.embed_read_input_type,
            "truncate": self.embedding_config.embed_truncate,
        }
        self.pc.create_index_for_model(
            name=index_name,
            cloud=self.config.cloud,
            region=self.config.environment,
            embed={
                "model": model_name,
                "field_map": {"text": "chunk_text"},
                "write_parameters": write_params,
                "read_parameters": read_params,
            },
        )

    def _create_sparse_pinecone_index(self, index_name: str, model_name: str) -> None:
        """Create sparse Pinecone index with max_tokens_per_sequence configured."""
        logger.info(
            f"Creating sparse index '{index_name}' with model: {model_name}, "
            f"max_tokens_per_sequence: {self.embedding_config.sparse_max_tokens}"
        )
        if self.pc is None:
            raise RuntimeError("Pinecone client not initialized")
        write_params = {
            "input_type": self.embedding_config.embed_write_input_type,
            "truncate": self.embedding_config.embed_truncate,
            "max_tokens_per_sequence": self.embedding_config.sparse_max_tokens,
        }
        read_params = {
            "input_type": self.embedding_config.embed_read_input_type,
            "truncate": self.embedding_config.embed_truncate,
            "max_tokens_per_sequence": self.embedding_config.sparse_max_tokens,
        }
        self.pc.create_index_for_model(
            name=index_name,
            cloud=self.config.cloud,
            region=self.config.environment,
            embed={
                "model": model_name,
                "field_map": {"text": "chunk_text"},
                "write_parameters": write_params,
                "read_parameters": read_params,
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

    def _ensure_namespace_schema(
        self, namespace: str, schema_fields: List[str]
    ) -> None:
        """Create a namespace with a filterable metadata schema on both indexes if absent.

        The namespace creation API (2025-10) accepts a schema that marks metadata
        fields as filterable. This is called once per upsert session, before the
        first batch is written. The call is idempotent: if the namespace already
        exists it is skipped.

        Args:
            namespace: The Pinecone namespace name.
            schema_fields: Metadata field names to mark as filterable.
        """
        if not namespace or not schema_fields:
            return

        schema_payload = {"fields": {f: {"filterable": True} for f in schema_fields}}

        for index in (self.dense_index, self.sparse_index):
            if index is None:
                continue
            try:
                existing_names = {ns.name for ns in index.list_namespaces()}
                if namespace in existing_names:
                    logger.debug(
                        "Namespace '%s' already exists on index '%s', skipping schema creation",
                        namespace,
                        getattr(index, "_index_name", "unknown"),
                    )
                    continue

                index_host = index._config.host
                # Strip any scheme prefix — index_host may already be "https://..."
                host_clean = index_host.removeprefix("https://").removeprefix("http://")
                response = requests.post(
                    f"https://{host_clean}/namespaces",
                    headers={
                        "Api-Key": self._active_api_key,
                        "X-Pinecone-Api-Version": "2025-10",
                        "Content-Type": "application/json",
                    },
                    json={"name": namespace, "schema": schema_payload},
                    timeout=30,
                )
                if response.status_code == 409:
                    logger.debug(
                        "Namespace '%s' already exists (409), skipping", namespace
                    )
                elif not response.ok:
                    logger.warning(
                        "Failed to create namespace '%s' with schema: %s %s",
                        namespace,
                        response.status_code,
                        response.text,
                    )
                else:
                    logger.info(
                        "Created namespace '%s' with schema fields: %s",
                        namespace,
                        schema_fields,
                    )
            except Exception as e:
                logger.warning(
                    "Could not create namespace '%s' with schema: %s", namespace, e
                )

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

    def process_documents(
        self,
        documents: List[Document],
        namespace: Optional[str] = None,
        is_chunked: bool = False,
        process_type: ProcessType | str = ProcessType.UPSERT,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process documents with one of supported process types.

        Supported process types:
        - upsert
        - update_metadata_by_id
        - update_document_by_id
        - update_value_by_id
        """
        try:
            process_mode = ProcessType.from_value(process_type)
            if not documents:
                logger.warning("No documents to process")
                return {
                    "processed": 0,
                    "total": 0,
                    "errors": [],
                    "failed_documents": [],
                    "process_type": process_mode.value,
                }

            self._ensure_indexes_ready()
            records = self._prepare_records_to_process(documents, is_chunked)

            if process_mode == ProcessType.UPSERT:
                if records:
                    schema_fields = sorted(
                        {
                            k
                            for record in records
                            for k in record
                            if k not in ("id", "chunk_text")
                        }
                    )
                    self._ensure_namespace_schema(namespace, schema_fields)
                total_upserted, errors, failed_documents = self._upsert_records(
                    records, namespace
                )
                return self._build_process_result(
                    total_processed=total_upserted,
                    total=len(documents),
                    errors=errors,
                    failed_documents=failed_documents,
                    process_type=process_mode,
                )

            if process_mode == ProcessType.UPDATE_METADATA_BY_ID:
                return self._update_by_id(
                    records=records,
                    namespace=namespace,
                    process_type=ProcessType.UPDATE_METADATA_BY_ID,
                    keys=keys,
                )
            if process_mode == ProcessType.UPDATE_DOCUMENT_BY_ID:
                return self._update_by_id(
                    records=records,
                    namespace=namespace,
                    process_type=ProcessType.UPDATE_DOCUMENT_BY_ID,
                )
            if process_mode == ProcessType.UPDATE_VALUE_BY_ID:
                return self._update_by_id(
                    records=records,
                    namespace=namespace,
                    process_type=ProcessType.UPDATE_VALUE_BY_ID,
                )

            raise ValueError(f"Unhandled process_type: {process_mode.value}")

        except Exception as e:
            logger.error(
                "Error in process_documents (type=%s): %s",
                process_type,
                e,
            )
            raise

    def _mark_batch_failed(
        self, batch_records: List[Dict[str, Any]], error: Exception, start_idx: int
    ) -> List[Dict[str, Any]]:
        """Mark batch documents as failed."""
        failed = []
        for idx, doc in enumerate(batch_records):
            failed.append(
                {
                    "doc_id": doc.get("id", doc.get("url", f"doc_{start_idx}_{idx}")),
                    "type": doc.get("type", "unknown"),
                    "reason": f"Batch upsert failed: {str(error)}",
                    "metadata": {
                        k: v for k, v in doc.items() if k not in ("id", "chunk_text")
                    },
                }
            )
        return failed

    def _upsert_records(
        self,
        documents: List[Document],
        namespace: Optional[str],
    ) -> tuple[int, List[str], List[Dict[str, Any]]]:
        """Upsert all batches."""
        total_upserted, errors, failed_docs = 0, [], []
        total_skipped_existing = 0
        batch_size = self.config.batch_size
        batch_start_idxs = list(range(0, len(documents), batch_size))
        for batch_idx, i in enumerate(batch_start_idxs):
            batch = documents[i : i + batch_size]
            batch_num = batch_idx + 1
            try:
                filtered_batch, skipped_existing = self._filter_existing_records(
                    batch, namespace
                )
                total_skipped_existing += skipped_existing

                if not filtered_batch:
                    logger.info(
                        "Skipped batch %s/%s: %s records already exist",
                        batch_num,
                        len(batch_start_idxs),
                        skipped_existing,
                    )
                    continue

                self._upsert_batch(filtered_batch, namespace, batch_num)
                total_upserted += len(filtered_batch)
                logger.info(
                    "Upserted batch %s/%s: %s documents (skipped existing: %s)",
                    batch_num,
                    len(batch_start_idxs),
                    len(filtered_batch),
                    skipped_existing,
                )

            except Exception as e:
                error_msg = f"Error upserting batch {batch_num}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                failed_docs.extend(self._mark_batch_failed(filtered_batch, e, i))

        if total_skipped_existing > 0:
            logger.info(
                "Skipped %s existing records during upsert",
                total_skipped_existing,
            )

        return total_upserted, errors, failed_docs

    def _extract_existing_ids_from_fetch_response(
        self, fetch_response: Any
    ) -> set[str]:
        """Normalize fetch response and return existing record IDs."""
        if fetch_response is None:
            return set()

        # SDK responses may expose vectors either as an attribute or as a dict key.
        vectors = getattr(fetch_response, "vectors", None)
        if vectors is None and isinstance(fetch_response, dict):
            vectors = fetch_response.get("vectors")
        if not vectors:
            return set()

        # vectors is commonly a dict[id, record], but keep fallback for list-like values.
        if isinstance(vectors, dict):
            return {str(record_id) for record_id in vectors.keys()}

        existing_ids: set[str] = set()
        try:
            for item in vectors:
                record_id = getattr(item, "id", None)
                if record_id is None and isinstance(item, dict):
                    record_id = item.get("id")
                if record_id is not None:
                    existing_ids.add(str(record_id))
        except TypeError:
            return set()
        return existing_ids

    def _filter_existing_records(
        self,
        records: List[Dict[str, Any]],
        namespace: Optional[str],
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Remove records that already exist in the dense index.

        This prevents unnecessary re-embedding and duplicate upsert writes when
        the same document IDs are processed again.
        """
        if not records:
            return records, 0

        self._ensure_indexes_ready()
        record_ids = [str(r["id"]) for r in records if "id" in r]
        if not record_ids:
            return records, 0

        try:
            fetch_response = self.dense_index.fetch(ids=record_ids, namespace=namespace)
            existing_ids = self._extract_existing_ids_from_fetch_response(
                fetch_response
            )
            if not existing_ids:
                return records, 0

            filtered = [r for r in records if str(r.get("id")) not in existing_ids]
            return filtered, len(records) - len(filtered)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Do not block ingestion if existence checks fail unexpectedly.
            logger.warning(
                "Existing record check failed; proceeding with full upsert: %s",
                e,
            )
            return records, 0

    def _build_process_result(
        self,
        total_processed: int,
        total: int,
        errors: List[str],
        failed_documents: List[Dict[str, Any]],
        process_type: ProcessType,
    ) -> Dict[str, Any]:
        """Build process result."""
        result = {
            "processed": total_processed,
            "upserted": total_processed if process_type == ProcessType.UPSERT else 0,
            "total": total,
            "errors": errors,
            "failed_documents": failed_documents,
            "failed_count": len(failed_documents),
            "process_type": process_type.value,
        }
        logger.info(
            f"Process complete ({process_type.value}): {result['processed']}/{result['total']} documents, "
            f"{result['failed_count']} failed"
        )
        if failed_documents:
            logger.warning(
                f"Failed documents summary: {result['failed_count']} documents failed to process"
            )
        return result

    def _ensure_indexes_ready(self) -> None:
        """Ensure indexes are ready."""
        if not self._dense_index_initialized or not self._sparse_index_initialized:
            self._get_or_create_indexes()
        if self.dense_index is None or self.sparse_index is None:
            raise RuntimeError("Pinecone indexes not initialized")

    def _prepare_records_to_process(
        self, documents: List[Document], is_chunked: bool
    ) -> List[Dict[str, Any]]:
        """Prepare batch records for upsert."""
        records = []
        chunked_documents = (
            self.text_splitter.split_documents(documents)
            if not is_chunked
            else documents
        )

        for doc in tqdm(
            chunked_documents,
            desc="Preparing records",
            leave=False,
        ):
            text = doc.page_content.strip() if doc.page_content else ""
            if not self._is_valid_chunk(
                text, self.config.min_text_length, self.config.min_words
            ):
                continue

            metadata = doc.metadata or {}
            if "author" in metadata:
                if isinstance(metadata["author"], list):
                    text = f"Author: {', '.join(metadata['author'])}\n\n{text}"
                elif metadata["author"] != "":
                    text = f"Author: {metadata['author']}\n\n{text}"

            if "title" in metadata and metadata["title"] != "":
                text = f"Title: {metadata['title']}\n\n{text}"
            elif "subject" in metadata and metadata["subject"] != "":
                text = f"Subject: {metadata['subject']}\n\n{text}"

            original_doc_id = metadata.get("doc_id", metadata.get("url"))
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
        """Upsert batch to both indexes. On sparse failure, rolls back dense to avoid partial state."""
        self._ensure_indexes_ready()
        record_ids = [r.get("id") for r in records if r.get("id")]
        try:
            self.dense_index.upsert_records(records=records, namespace=namespace)
        except Exception as e:
            logger.error(
                "Failed to upsert batch %s to dense index: %s. Records: %s",
                batch_num,
                e,
                record_ids,
            )
            raise
        try:
            self.sparse_index.upsert_records(records=records, namespace=namespace)
        except Exception as e:
            logger.error(
                "Sparse upsert failed for batch %s, rolling back dense upsert: %s",
                batch_num,
                e,
            )
            if record_ids and namespace:
                try:
                    self.dense_index.delete(ids=record_ids, namespace=namespace)
                    logger.info(
                        "Rolled back %s records from dense index after sparse failure",
                        len(record_ids),
                    )
                except Exception as rollback_e:
                    logger.error(
                        "Rollback of dense index failed after sparse upsert error: %s",
                        rollback_e,
                    )
            raise

    def _update_to_index(
        self,
        index: Any,
        doc_id: str,
        namespace: Optional[str],
        index_type: str,
        values: Optional[List[float]] = None,
        sparse_values: Optional[Dict[str, Any]] = None,
        set_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a single record in one index by ID."""
        update_kwargs: Dict[str, Any] = {"id": doc_id, "namespace": namespace}
        if values is not None:
            update_kwargs["values"] = values
        if sparse_values is not None:
            update_kwargs["sparse_values"] = sparse_values
        if set_metadata is not None:
            update_kwargs["set_metadata"] = set_metadata

        try:
            index.update(**update_kwargs)
        except Exception as e:
            logger.error(f"Failed to update id={doc_id} in {index_type} index: {e}")
            raise

    def _resolve_target_id(self, metadata: Dict[str, Any], fallback_idx: int) -> str:
        """Resolve target vector ID from metadata."""
        return str(
            metadata.get("id")
            or metadata.get("vector_id")
            or metadata.get("pinecone_id")
            or f"doc_{fallback_idx}"
        )

    def _extract_metadata_update(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata patch payload."""
        if isinstance(metadata.get("metadata_update"), dict):
            return metadata["metadata_update"]
        if isinstance(metadata.get("set_metadata"), dict):
            return metadata["set_metadata"]

        reserved_keys = {
            "id",
            "vector_id",
            "pinecone_id",
            "values",
            "sparse_values",
            "chunk_text",
        }
        return {k: v for k, v in metadata.items() if k not in reserved_keys}

    def _update_by_id(
        self,
        records: List[Dict[str, Any]],
        namespace: Optional[str],
        keys: Any = None,
        process_type: ProcessType = ProcessType.UPDATE_DOCUMENT_BY_ID,
    ) -> Dict[str, Any]:
        """Update metadata only for existing vectors by ID."""
        if not records:
            return self._build_process_result(
                total_processed=0,
                total=0,
                errors=[],
                failed_documents=[],
                process_type=process_type,
            )
        processed, errors, failed_documents = 0, [], []
        selected_keys = []
        if process_type == ProcessType.UPDATE_DOCUMENT_BY_ID:
            selected_keys = [k for k in records[0].keys() if k != "id"]
        elif process_type == ProcessType.UPDATE_METADATA_BY_ID:
            selected_keys = keys or [
                k for k in records[0].keys() if k not in ("id", "chunk_text")
            ]
        elif process_type == ProcessType.UPDATE_VALUE_BY_ID:
            selected_keys = ["chunk_text"]
        else:
            raise ValueError(f"Unsupported process type: {process_type}")

        for record in tqdm(
            records,
            desc=f"Updating by id ({process_type.value})",
            leave=False,
        ):
            doc_id = record["id"]
            update_metadata = {}
            for key in selected_keys:
                if key not in record:
                    continue
                update_metadata[key] = record[key]
            if not update_metadata:
                missing = [k for k in selected_keys if k not in record]
                failed_documents.append(
                    {
                        "doc_id": doc_id,
                        "type": record.get("type", "unknown"),
                        "reason": f"Keys not found in record: {missing}",
                    }
                )
                continue
            try:
                dense_kwargs: Dict[str, Any] = {
                    "id": doc_id,
                    "namespace": namespace,
                    "set_metadata": update_metadata,
                }
                sparse_kwargs: Dict[str, Any] = {
                    "id": doc_id,
                    "namespace": namespace,
                    "set_metadata": update_metadata,
                }
                if process_type == ProcessType.UPDATE_VALUE_BY_ID:
                    if record.get("values") is not None:
                        dense_kwargs["values"] = record["values"]
                    if record.get("sparse_values") is not None:
                        sparse_kwargs["sparse_values"] = record["sparse_values"]
                self.dense_index.update(**dense_kwargs)
                self.sparse_index.update(**sparse_kwargs)
                processed += 1
            except Exception as e:
                error_msg = f"Error updating metadata for id={doc_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                failed_documents.append(
                    {
                        "doc_id": record.get("doc_id", record.get("url")),
                        "type": record.get("type", "unknown"),
                        "reason": error_msg,
                        "text_length": len(record.get("chunk_text", "")),
                    }
                )

        return self._build_process_result(
            total_processed=processed,
            total=len(records),
            errors=errors,
            failed_documents=failed_documents,
            process_type=process_type,
        )

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
        """Format index statistics and compute per-namespace dense vs sparse deltas."""
        dense_ns = dense_stats.get("namespaces", {}) or {}
        sparse_ns = sparse_stats.get("namespaces", {}) or {}

        def _vector_count(ns_dict: Dict[str, Any], name: str) -> int:
            entry = ns_dict.get(name)
            if entry is None:
                return 0
            if isinstance(entry, dict):
                return int(entry.get("vector_count", 0) or 0)
            return int(getattr(entry, "vector_count", 0) or 0)

        all_names = set(dense_ns) | set(sparse_ns)
        namespace_deltas: List[Dict[str, Any]] = []
        for name in sorted(all_names):
            d = _vector_count(dense_ns, name)
            s = _vector_count(sparse_ns, name)
            delta = d - s
            namespace_deltas.append(
                {"namespace": name, "dense": d, "sparse": s, "delta": delta}
            )

        mismatched = [x for x in namespace_deltas if x["delta"] != 0]

        return {
            "dense_index": {
                "total_vectors": dense_stats.get("total_vector_count", 0),
                "dimension": dense_stats.get("dimension", 0),
                "index_fullness": dense_stats.get("index_fullness", 0),
                "namespaces": dense_ns,
            },
            "sparse_index": {
                "total_vectors": sparse_stats.get("total_vector_count", 0),
                "dimension": sparse_stats.get("dimension", 0),
                "index_fullness": sparse_stats.get("index_fullness", 0),
                "namespaces": sparse_ns,
            },
            "namespace_deltas": namespace_deltas,
            "mismatched_namespaces": mismatched,
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
            "namespace_deltas": [],
            "mismatched_namespaces": [],
        }
