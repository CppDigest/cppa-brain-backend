"""
Pinecone query interface for hybrid search retrieval.

Pure Pinecone query class that performs hybrid search (dense + sparse)
with reranking, without any LLM dependencies.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from pinecone import Pinecone
    from langchain_core.documents import Document
except ImportError as e:
    Pinecone = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

from config import PineconeConfig

logger = logging.getLogger(__name__)


class PineconeQuery:
    """
    Pure Pinecone query interface for hybrid search retrieval.

    Loads configuration from environment variables and directly queries Pinecone
    indexes for hybrid search (dense + sparse) with reranking.
    No LLM dependencies - only retrieval.
    """

    def __init__(
        self,
        pinecone_config: Optional[PineconeConfig] = None,
    ):
        self._validate_imports()

        # Load config from environment variables (create new instances if not provided)
        self.pinecone_config = pinecone_config or PineconeConfig()

        # Initialize components
        self.pc: Optional[Pinecone] = None
        self.dense_index: Optional[Any] = None
        self.sparse_index: Optional[Any] = None
        self._indexes_initialized = False

    def _validate_imports(self) -> None:
        """Validate that required imports are available."""
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Missing optional dependencies required for PineconeQuery. "
                "Install with: pip install pinecone-client langchain-core"
            ) from _IMPORT_ERROR

    def _initialize_pinecone_client(self) -> None:
        """Initialize Pinecone client if not already initialized."""
        if self.pc is None:
            self.pc = Pinecone(api_key=self.pinecone_config.api_key)
            logger.info("Pinecone client initialized")

    def _ensure_indexes_ready(self) -> None:
        """Ensure Pinecone indexes are initialized and ready."""
        if self._indexes_initialized:
            return

        self._initialize_pinecone_client()
        if self.pc is None:
            raise RuntimeError("Pinecone client not initialized")

        dense_name = self.pinecone_config.index_name
        sparse_name = f"{self.pinecone_config.index_name}-sparse"

        self.dense_index = self.pc.Index(dense_name)
        self.sparse_index = self.pc.Index(sparse_name)
        self._indexes_initialized = True
        logger.info(f"Connected to indexes: {dense_name} and {sparse_name}")

    def _search_index(
        self,
        index: Any,
        query: str,
        top_k: int,
        namespace: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search a Pinecone index using text query."""
        try:
            # Build query dict - only include filter if it's not None
            query_dict = {
                "top_k": top_k,
                "inputs": {"text": query},
            }

            # Only add filter if metadata_filter is provided and not None
            if metadata_filter is not None:
                query_dict["filter"] = metadata_filter

            result = index.search(
                namespace=namespace,
                query=query_dict,
            )
            return result.get("result", {}).get("hits", [])
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []

    def _merge_results(
        self, dense_hits: List[Dict[str, Any]], sparse_hits: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from dense and sparse searches."""
        deduped: Dict[str, Dict[str, Any]] = {}
        for hit in dense_hits + sparse_hits:
            hit_id = hit.get("_id", "")
            hit_score = hit.get("_score", 0.0)

            if hit_id in deduped and deduped[hit_id].get("_score", 0.0) >= hit_score:
                continue

            hit_metadata = {}
            content = ""
            for key, metadata in hit.get("fields", {}).items():
                if key == "chunk_text":
                    content = metadata
                else:
                    hit_metadata[key] = metadata
            deduped[hit_id] = {
                "_id": hit_id,
                "_score": hit_score,
                "chunk_text": content,
                "metadata": hit_metadata,
            }

        sorted_hits = sorted(
            deduped.values(), key=lambda x: x.get("_score", 0.0), reverse=True
        )

        return sorted_hits

    def _rerank_results(
        self, query: str, results: List[Dict[str, Any]], top_n: int
    ) -> List[Document]:
        """Rerank results using Pinecone's reranking model."""
        if not results or self.pc is None:
            return []

        try:
            rerank_result = self.pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=results,
                rank_fields=["chunk_text"],
                top_n=top_n,
                return_documents=True,
                parameters={"truncate": "END"},
            )

            documents = self._convert_to_documents(rerank_result)
            return documents

        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return [
                Document(
                    page_content=result.get("chunk_text", ""),
                    metadata={"id": result.get("_id", ""), "score": 0.0},
                )
                for result in results[:top_n]
            ]

    def _convert_to_documents(self, rerank_result: Any) -> List[Document]:
        """Convert rerank result to documents."""
        documents: List[Document] = []
        for item in rerank_result.data:
            document = item.get("document", {})
            doc_id = document.get("_id", "")
            chunk_text = document.get("chunk_text", "")
            score = item.get("score", 0.0)
            metadata = document.get("metadata", {})
            metadata["score"] = float(score)
            metadata["reranked"] = True

            doc = Document(
                id=doc_id,
                page_content=chunk_text,
                metadata=metadata,
            )
            documents.append(doc)
        return documents

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        namespace: str = "mailing",
        use_reranking: bool = True,
    ) -> List[Document]:
        """
        Query Pinecone indexes using hybrid search (dense + sparse) with optional reranking.

        Args:
            query: Query text string
            top_k: Number of documents to retrieve (defaults to config value)
            metadata_filter: Optional metadata filter
            namespace: Pinecone namespace
            use_reranking: Whether to use reranking (default: True)

        Returns:
            List of retrieved Document objects with similarity scores
        """
        top_k = top_k or self.pinecone_config.top_k

        # Ensure indexes are ready
        self._ensure_indexes_ready()
        if self.dense_index is None or self.sparse_index is None:
            raise RuntimeError("Pinecone indexes not initialized")

        # Perform hybrid search
        dense_hits = self._search_index(
            self.dense_index, query, top_k, namespace, metadata_filter
        )
        sparse_hits = self._search_index(
            self.sparse_index, query, top_k, namespace, metadata_filter
        )

        # Merge results
        merged_results = self._merge_results(dense_hits, sparse_hits)

        # Optionally rerank
        if use_reranking:
            documents = self._rerank_results(query, merged_results, top_n=top_k)
        else:
            # Return without reranking
            documents = [
                Document(
                    page_content=result.get("chunk_text", ""),
                    metadata={"id": result.get("_id", ""), "score": 0.0},
                )
                for result in merged_results[:top_k]
            ]

        logger.info(
            f"Retrieved {len(documents)} documents from hybrid search "
            f"(dense: {len(dense_hits)}, sparse: {len(sparse_hits)})"
        )
        return documents
