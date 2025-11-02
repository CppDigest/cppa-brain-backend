"""
Hybrid retriever for LangChain RAG pipeline (Dense + Sparse only)
"""

import os
import pickle
from typing import List
from datetime import datetime
import math
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import hashlib
import gc

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LangChainHybridRetriever:
    """Hybrid retriever combining dense (vector) and sparse (BM25) retrieval for LangChain"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chroma_persist_dir: str = "./chroma_db",
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        final_top_k: int = 50,
        force_reindex: bool = False,
        half_life: int = 1500,
    ):
        self.documents = []
        self.embedding_model_name = embedding_model
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.force_reindex = force_reindex
        self.chroma_persist_dir = chroma_persist_dir
        self.logger = logger.bind(name="LangChainHybridRetriever")
        self.vector_store = None
        self.dense_retriever = None
        self.half_life = half_life
        # Initialize components
        try:
            self._setup_embedding_model()
        except Exception as e:
            self.logger.exception("Failed to initialize LangChainHybridRetriever: %s", e)
            raise

    def reindex(self, documents: List[Document], status: dict = None):
        """Reindex the hybrid retriever - create new indices from documents"""
        if status is None:
            status = {
                "vector_store": False,
                "sparse_retriever": False,
            }
        if all(status.values()):
            return

        try:
            self.logger.info("Starting reindex with %d documents", len(documents))
            self.documents = documents

            # Force reindex by clearing existing data
            self._clear_existing_indices(status)

            # Create new indices
            if not status["vector_store"]:
                status["vector_store"] = self._create_vector_store()
            if not status["sparse_retriever"]:
                status["sparse_retriever"] = self._create_sparse_retriever()

            self.logger.info("Reindex completed successfully")
        except Exception as e:
            self.logger.exception("Failed to reindex: %s", e)

    def load_index(self) -> dict:
        """Load the hybrid retriever index - load from saved data"""
        status = {
            "vector_store": False,
            "sparse_retriever": False,
        }

        try:
            self.logger.info(
                "Loading existing indices from %s", self.chroma_persist_dir
            )
            status["vector_store"] = self._load_vector_store()
            status["sparse_retriever"] = self._load_sparse_retriever()
        except Exception as e:
            self.logger.exception("Failed to load indices: %s", e)

        return status

    def _clear_existing_indices(self, status: dict):
        """Clear existing indices for reindexing"""
        try:
            self.logger.info("Clearing existing indices")
            if not status["vector_store"]:
                self._clear_vector_store()
            if not status["sparse_retriever"]:
                self._clear_sparse_retriever()
        except Exception as e:
            self.logger.exception("Failed to clear indices: %s", e)
            raise

    def _clear_vector_store(self):
        """Clear ChromaDB vector store"""
        try:
            self.logger.info("Clearing vector store")
            # ChromaDB clears on recreate
            self.logger.info("Vector store ready for reindex")
        except Exception as e:
            self.logger.info("No existing vector store to clear: %s", e)

    def _clear_sparse_retriever(self):
        """Clear TF-IDF vectorizer files for reindexing"""
        try:
            self.logger.info("Clearing TF-IDF vectorizer")
            # Only TF-IDF vectorizer is persisted (BM25 created dynamically)
            tfidf_file = os.path.join(self.chroma_persist_dir, "tfidf_vectorizer.pkl")
            if os.path.exists(tfidf_file):
                os.remove(tfidf_file)
                self.logger.info("Deleted tfidf_vectorizer.pkl")

            self.logger.info("TF-IDF vectorizer cleared")
        except Exception as e:
            self.logger.warning("Error clearing TF-IDF vectorizer: %s", e)

    def _setup_embedding_model(self):
        """Setup embedding model"""
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cuda" if self._is_cuda_available() else "cpu"},
            )
            self.logger.info(
                "Embedding model initialized: %s", self.embedding_model_name
            )
        except Exception as e:
            self.logger.exception("Failed to initialize embedding model: %s", e)
            raise

    def _create_vector_store(self) -> bool:
        """Create new vector store with documents"""
        try:
            # Create embedded ChromaDB vector store
            self.logger.info("Creating embedded Chroma vector store")
            self.vector_store = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embedding_model,
                persist_directory=self.chroma_persist_dir,
            )
            self.logger.info("Embedded vector store created successfully")

            # Create dense retriever
            self.dense_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.dense_top_k,
                    "score_threshold": 0.3},
            )

            self.logger.info("Vector store and dense retriever initialized successfully")
            return True
        except Exception as e:
            self.logger.exception("Failed to create vector store: %s", e)
            return False

    def _load_vector_store(self) -> bool:
        """Load existing vector store"""
        try:
            self.logger.info("Loading embedded vector store")
            
            # Check if ChromaDB directory exists and has data
            if not os.path.exists(self.chroma_persist_dir):
                self.logger.info("ChromaDB directory does not exist: %s", self.chroma_persist_dir)
                return False
            
            # Check for chroma.sqlite3 or collection data
            chroma_db_file = os.path.join(self.chroma_persist_dir, "chroma.sqlite3")
            if not os.path.exists(chroma_db_file):
                self.logger.info("ChromaDB data file not found: %s", chroma_db_file)
                return False
            
            # Load existing ChromaDB
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.chroma_persist_dir,
            )
            
            # Get documents from vector store
            self.documents = []  # Will be populated during retrieval

            # Create dense retriever
            self.dense_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.dense_top_k,
                    "score_threshold": 0.3
                    },
            )

            self.logger.info("Vector store loaded successfully")
            gc.collect()  # Clean up
            return True
        except Exception as e:
            self.logger.exception("Failed to load vector store: %s", e)
            return False

    def _create_sparse_retriever(self) -> bool:
        """Create and save TF-IDF vectorizer for reranking (BM25 created dynamically)"""
        try:
            self.logger.info("Creating TF-IDF vectorizer for reranking")
            
            # Prepare documents for TF-IDF
            doc_texts = [doc.page_content for doc in self.documents]
            
            # Setup TF-IDF for reranking
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000, stop_words="english", ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
            
            # Save TF-IDF vectorizer
            os.makedirs(self.chroma_persist_dir, exist_ok=True)
            with open(os.path.join(self.chroma_persist_dir, "tfidf_vectorizer.pkl"), "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)

            self.logger.info("TF-IDF vectorizer created and saved successfully")
            return True
        except Exception as e:
            self.logger.exception("Failed to create TF-IDF vectorizer: %s", e)
            return False

    def _load_sparse_retriever(self) -> bool:
        """Load TF-IDF vectorizer for reranking (BM25 created dynamically during retrieval)"""
        try:
            self.logger.info("Loading TF-IDF vectorizer")
            
            # Load TF-IDF vectorizer for reranking
            with open(os.path.join(self.chroma_persist_dir, "tfidf_vectorizer.pkl"), "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

            self.logger.info("TF-IDF vectorizer loaded successfully")
            return True
        except Exception as e:
            self.logger.exception("Failed to load TF-IDF vectorizer: %s", e)
            return False

    def retrieve(self, query: str, fetch_k: int = 10, filter_types: List[str] = None) -> List[Document]:
        """Perform hybrid retrieval (Dense + Sparse) with optional type filtering
        
        Args:
            query: Search query
            fetch_k: Number of results to return
            filter_types: List of document types to filter by (e.g., ["documentation", "mail", "slack"])
                         If None or empty, no filtering is applied
        
        Returns:
            List of Document objects matching the query and filters
        """
        # Dense retrieval
        if filter_types is None: 
            dense_docs = self.dense_retriever.invoke(query)
        elif len(filter_types) == 0 or len(filter_types) == 2:
            dense_docs = self.dense_retriever.invoke(query)
        elif len(filter_types) == 1:
            filters = {"type": filter_types[0]}
            dense_docs = self.dense_retriever.invoke(query, filter=filters)
        else:
            filters = []
            for filter_type in filter_types:
                filters.append({"type": filter_type})
            if len(filters) > 1:
                filters = {"$or": filters}
            else:
                filters = filters[0]
            dense_docs = self.dense_retriever.invoke(query, filter=filters)

        # dense_docs = self.dense_retriever.invoke(query)
        
        # sparse_docs = self.sparse_retriever.invoke(query)
        sparse_retriever = BM25Retriever.from_documents(
                documents=dense_docs,
                k=self.sparse_top_k
            )

        # Sparse retrieval
        sparse_docs = sparse_retriever.invoke(query)

        # Combine and rerank results
        combined_docs = self._combine_results(dense_docs, sparse_docs)

        # Filter by document type if specified
        if filter_types and len(filter_types) > 0:
            # Normalize filter types to lowercase for comparison
            filter_types_normalized = [ft.lower() for ft in filter_types]
            filtered_docs = []
            for doc in combined_docs:
                doc_type = doc.metadata.get("type", "documentation").lower()
                if doc_type in filter_types_normalized:
                    filtered_docs.append(doc)
            combined_docs = filtered_docs
            self.logger.info(
                f"Filtered to {len(combined_docs)} documents matching types: {filter_types}"
            )

        # Rerank using TF-IDF similarity
        reranked_docs = self._rerank_results(query, combined_docs)
        fetch_k = fetch_k if fetch_k is not None else self.final_top_k
        fetch_k = min(fetch_k, len(reranked_docs))
        
        gc.collect()
        return reranked_docs[: fetch_k]

    def _combine_results(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:
        """Combine results from dense and sparse retrievers"""
        # Create a dictionary to store combined scores
        combined_scores = {}

        # Add dense retrieval scores
        for i, doc in enumerate(dense_docs):
            doc_id = doc.id if doc.id else id(doc.metadata["url"])
            # Higher score for earlier results
            combined_scores[doc_id] = {
                "document": doc,
                "dense_score": 1.0 / (i + 1),
                "sparse_score": 0.0,
            }

        # Add sparse retrieval scores
        for i, doc in enumerate(sparse_docs):
            doc_id = doc.id if doc.id else id(doc.metadata["url"])

            if doc_id in combined_scores:
                combined_scores[doc_id]["sparse_score"] = 1.0 / (i + 1)
            else:
                combined_scores[doc_id] = {
                    "document": doc,
                    "dense_score": 0.0,
                    "sparse_score": 1.0 / (i + 1),
                }

        # Create combined documents with ensemble scoring
        combined_docs = []
        for doc_id, scores in combined_scores.items():
            # Weighted ensemble scoring (60% dense, 40% sparse)
            ensemble_score = (
                0.6 * scores["dense_score"]
                + 0.4 * scores["sparse_score"]
            )

            # Update document metadata with ensemble score
            doc = scores["document"]
            doc.metadata["ensemble_score"] = ensemble_score
            
            combined_docs.append(doc)

        return combined_docs

    def _rerank_results(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank results using query-document similarity"""
        # Use TF-IDF similarity for reranking
        query_vector = self.tfidf_vectorizer.transform([query])

        reranked_docs = []
        for doc in docs:
            doc_text = doc.page_content
            doc_vector = self.tfidf_vectorizer.transform([doc_text])
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            # similarity = 0.5

            # Combine original score with TF-IDF similarity
            original_score = doc.metadata.get("ensemble_score", 0.0)
            final_score = 0.9 * original_score + 0.1 * similarity

            # Add time exponential bonus for mail documents 0~1.0
            if doc.metadata.get("type") == "mail" and "date" in doc.metadata:
                # Clean content without modifying original unnecessarily
                cleaned_content = doc_text.replace(">>", "")
                if cleaned_content != doc_text:
                    doc.page_content = cleaned_content
                
                time_bonus = self._calculate_time_bonus(doc.metadata["date"])
                final_score = final_score * (0.5 + time_bonus)

            # Update document metadata
            doc.metadata["final_score"] = final_score
            
            # Remove temporary metadata (don't delete "source" - it's useful for tracking)
            if "ensemble_score" in doc.metadata:
                del doc.metadata["ensemble_score"]
            
            reranked_docs.append(doc)
            

        # Sort by final score
        reranked_docs.sort(
            key=lambda x: x.metadata.get("final_score", 0.0), reverse=True
        )
        return reranked_docs

    def _calculate_time_bonus(self, date_str: str) -> float:
        """Calculate exponential time bonus for mail documents based on recency"""
        try:
            # Parse the date string (assuming ISO format or common formats)
            # Try multiple date formats
            date_formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 format
                "%a, %d %b %Y %H:%M:%S",
            ]
            
            doc_date = None
            for fmt in date_formats:
                try:
                    doc_date = datetime.strptime(date_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if doc_date is None:
                # If parsing fails, return no bonus
                self.logger.warning("Failed to parse date: %s", date_str)
                return 0.0
            
            # Remove timezone info if present for comparison
            if doc_date.tzinfo is not None:
                doc_date = doc_date.replace(tzinfo=None)
            
            # Calculate days since the email
            current_date = datetime.now()
            days_ago = (current_date - doc_date).days
            
            # Exponential decay: more recent = higher bonus
            # Bonus decays with half-life of ~365 days (1 year)
            # Recent emails (< 1 year) get significant bonus
            # Older emails get diminishing bonus
            
            # decay_rate = 0.002  # Adjust this to control decay speed
            # time_bonus = math.exp(-decay_rate * days_ago)
            
            time_bonus = math.exp(-days_ago / self.half_life)
            
            time_bonus = max(0.1, min(1.0, time_bonus))
            
            return time_bonus
            
        except Exception as e:
            self.logger.warning("Error calculating time bonus: %s", e)
            return 0.0

    def document_exists(self, doc_url: str) -> bool:
        """Check if a document with the given ID exists in the vector store
        
        Args:
            doc_url: Document URL to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            doc_id = self.get_document_id(doc_url)

            # Query ChromaDB for the document ID
            results = self.vector_store.get_by_ids([doc_id])
            return len(results) > 0
        except Exception as e:
            self.logger.warning("Error checking document existence for URL %s: %s", doc_url, e)
            return False
    
    def get_document_id(self, doc_url: str) -> str:
        """Get the document ID for a given URL"""
        return hashlib.md5(doc_url.encode()).hexdigest() + "-000"
    
    def check_existing_documents(self, documents: List[Document]) -> tuple[List[Document], List[Document]]:
        """Check which documents already exist in the database
        
        Args:
            documents: List of documents to check
            
        Returns:
            Tuple of (new_documents, existing_documents)
        """
        new_documents = []
        existing_documents = []
        
        for doc in documents:
            doc_url = doc.metadata.get('url', 'N/A')
            if self.document_exists(doc_url):
                existing_documents.append(doc)
                self.logger.debug("Document %s already exists (URL: %s)", doc.metadata.get('message_id', 'N/A'), doc_url)
            else:
                new_documents.append(doc)
        
        self.logger.info(
            "Checked %d documents: %d new, %d already exist",
            len(documents), len(new_documents), len(existing_documents)
        )
        
        return new_documents, existing_documents
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to vector store (BM25 created dynamically during retrieval)"""
        try:
            self.logger.info("Adding %d documents to hybrid retriever", len(documents))
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            self.logger.info("Documents added to vector store")
            
            # Note: BM25 is created dynamically during retrieval for memory efficiency
            # TF-IDF vectorizer will need to be retrained on next full reindex
            
            # Update documents list
            self.documents.extend(documents)
            self.logger.info("Total documents: %d", len(self.documents))
            
        except Exception as e:
            self.logger.exception("Failed to add documents: %s", e)
            raise
    
    def update_documents(self, updated_documents: List[Document]):
        """Update existing documents in both retrievers"""
        try:
            self.logger.info("Updating %d documents", len(updated_documents))
            document_ids = [doc.id for doc in updated_documents]
            # Delete old documents
            self.delete_documents(document_ids)
            
            # Add updated documents
            self.add_documents(updated_documents)
            
            self.logger.info("Documents updated successfully")
        except Exception as e:
            self.logger.exception("Failed to update documents: %s", e)
            raise
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents from vector store (BM25 created dynamically during retrieval)"""
        try:
            self.logger.info("Deleting %d documents", len(document_ids))
            
            # Delete from vector store (ChromaDB)
            try:
                self.vector_store.delete(ids=document_ids)
                self.logger.info("Documents deleted from vector store")
            except Exception as e:
                self.logger.warning("Could not delete from vector store: %s", e)
            
            # Note: BM25 is created dynamically during retrieval for memory efficiency
            # TF-IDF vectorizer will need to be retrained on next full reindex
            
            self.logger.info("Documents deleted successfully")
        except Exception as e:
            self.logger.exception("Failed to delete documents: %s", e)
            raise

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False
