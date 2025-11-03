"""
LangChain RAG Pipeline with hybrid retrieval
"""

import os
import sys
from typing import List, Dict, Any
import time
from tqdm import tqdm
from loguru import logger

sys.path.append(os.path.dirname(__file__))

from config.rag_config import LangChainConfig
from .hybrid_retriever import LangChainHybridRetriever
from .duckdb_retriever import DuckDBHybridRetriever
from .data_processor import BoostDataProcessor
from .caching_telemetry import (
    QueryCache,
    PerformanceTelemetry,
    CachedRetriever,
    InstrumentedRetriever,
)


class LangChainRAGPipeline:
    """High-performance RAG pipeline using LangChain with hybrid retrieval"""

    def __init__(self, config: LangChainConfig = None, **kwargs):
        # Use provided config or create from kwargs
        if config is None:
            config = LangChainConfig(**kwargs)

        self.config = config
        self.logger = logger.bind(name="LangChainRAGPipeline")

        # Validate configuration
        try:
            self.config.validate()
            self.logger.info("Configuration validated successfully")
        except Exception as e:
            self.logger.exception("Configuration validation failed: %s", e)
            raise

        # Initialize caching and telemetry
        self.cache = None
        self.telemetry = None
        if self.config.enable_cache:
            self.cache = QueryCache(
                cache_dir=self.config.cache_dir,
                ttl_seconds=self.config.cache_ttl_seconds,
                max_memory_entries=self.config.cache_max_memory_entries,
                max_disk_size_mb=self.config.cache_max_disk_size_mb,
                auto_cleanup_interval=self.config.cache_auto_cleanup_interval,
            )
            self.logger.info(
                f"Query cache initialized (TTL: {self.config.cache_ttl_seconds}s, "
                f"Max memory: {self.config.cache_max_memory_entries}, "
                f"Max disk: {self.config.cache_max_disk_size_mb}MB)"
            )

        if self.config.enable_telemetry:
            self.telemetry = PerformanceTelemetry(
                log_file=self.config.telemetry_log_file
            )
            self.logger.info("Performance telemetry initialized")

        # Initialize components
        try:
            self._setup_data_processor()
            self._setup_retriever()
            self.logger.info("LangChain RAG pipeline initialized successfully")
        except Exception as e:
            self.logger.exception("Failed to initialize LangChain RAG pipeline: %s", e)
            raise

    def _setup_data_processor(self):
        """Setup data processor"""
        try:
            self.data_processor = BoostDataProcessor(
                mail_data_dir=self.config.mail_data_dir,
                doc_data_dir=self.config.doc_data_dir,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            self.logger.info("Data processor initialized")
        except Exception as e:
            self.logger.exception("Failed to initialize data processor: %s", e)
            raise

    def _setup_retriever(self):
        """Setup hybrid retriever"""
        try:
            # Load and process documents
            os.environ["CHROMA_DISABLE_PERSISTENCE_CACHE"] = "1"
            base_retriever = LangChainHybridRetriever(
                embedding_model=self.config.embedding_model,
                chroma_persist_dir=self.config.chroma_persist_dir,
                dense_top_k=self.config.dense_top_k,
                sparse_top_k=self.config.sparse_top_k,
                final_top_k=self.config.final_top_k,
                force_reindex=self.config.force_reindex,
                half_life=self.config.half_life,
            )

            # Wrap with instrumentation and caching
            retriever = base_retriever
            if self.config.enable_telemetry and self.telemetry:
                retriever = InstrumentedRetriever(
                    retriever=retriever,
                    telemetry=self.telemetry,
                    stage_name="hybrid_retrieval",
                )
                self.logger.info("Retriever wrapped with telemetry")

            if self.config.enable_cache and self.cache:
                retriever = CachedRetriever(retriever=retriever, cache=self.cache)
                self.logger.info("Retriever wrapped with cache")

            self.hybrid_retriever = retriever
            self.base_retriever = base_retriever  # Keep reference to base for reindexing
        except Exception as e:
            self.logger.exception("Failed to setup hybrid retriever: %s", e)
            raise

        # Initialize status dictionary
        status = {
            "vector_store": False,
            "sparse_retriever": False,
        }

        if not self.config.force_reindex:
            try:
                status = self.base_retriever.load_index()
                if all(status.values()):
                    self.logger.info("Hybrid retriever index loaded successfully")
                    return
            except Exception as e:
                self.logger.exception("Failed to load hybrid retriever index: %s", e)
                self.logger.info("Force reindexing...")

        try:
            self.logger.info(
                f"Loading and processing documents from {self.config.mail_data_dir} and {self.config.doc_data_dir}"
            )
            documents = self.data_processor.load_documents()
            self.logger.info(f"Loaded {len(documents)} documents")
            self.logger.info("Chunking documents...")
            chunked_documents = self.data_processor.chunk_documents(documents)
            self.logger.info(f"Chunked {len(chunked_documents)} documents")
            # Pass status to reindex (will be None if force_reindex=True due to reset above)
            prev_status = None if self.config.force_reindex else status
            self.base_retriever.reindex(chunked_documents, prev_status)
        except Exception as e:
            self.logger.exception("Failed to load documents: %s", e)
            raise
    
    def retrieve(self, question: str, fetch_k: int = 10, filter_types: List[str] = None, str_results: bool = False):
        """Retrieve relevant documents with optional type filtering
        
        Args:
            question: Search query
            fetch_k: Number of results to return
            filter_types: List of document types to filter by (e.g., ["documentation", "mail", "slack"])
            str_results: If True, return text content instead of Document objects
        
        Returns:
            List of documents or text strings
        """
        retrieve_list = []
        try:
            # Get relevant documents with optional type filtering
            relevant_docs = self.hybrid_retriever.retrieve(question, fetch_k, filter_types)

            # Create context from documents
            if str_results:
                retrieve_list = [doc.page_content for doc in relevant_docs]
            else:
                retrieve_list = relevant_docs
        except Exception as e:
            self.logger.exception("Error during query: %s", e)
        return retrieve_list
        

    def query(self, question: str, fetch_k: int = 10, filter_types: List[str] = None) -> Dict[str, Any]:
        """Query the RAG pipeline with optional type filtering
        
        Args:
            question: Search query
            fetch_k: Number of results to return
            filter_types: List of document types to filter by (e.g., ["documentation", "mail", "slack"])
        
        Returns:
            Dictionary with answer, source documents, and metadata
        """
        start_time = time.time()
        self.logger.info("Processing query: %s (fetch_k=%d, filter_types=%s)", question, fetch_k, filter_types)

        # Start telemetry tracking
        if self.telemetry:
            self.telemetry.start_query(question)

        try:
            # Get relevant documents with optional type filtering
            if self.telemetry:
                self.telemetry.start_stage("document_retrieval")
            
            retrieve_list = self.retrieve(question, fetch_k, filter_types)
            
            if self.telemetry:
                self.telemetry.end_stage(
                    "document_retrieval",
                    {"fetch_k": fetch_k, "results": len(retrieve_list)}
                )

            # Prepare response (LLM generation disabled - retrieval only)
            response = {
                "answer": "",  # Empty since LLM is disabled
                "source_documents": retrieve_list,
                "query_time": time.time() - start_time,
                "retrieval_method": "hybrid",
            }

            # End telemetry tracking
            if self.telemetry:
                self.telemetry.end_query(results_count=len(retrieve_list), success=True)

            self.logger.info(
                "Query processed successfully in %.3fs", time.time() - start_time
            )
            return response

        except Exception as e:
            # End telemetry tracking with failure
            if self.telemetry:
                self.telemetry.end_query(results_count=0, success=False)
            
            self.logger.exception("Error during query: %s", e)
            return {
                "answer": "I apologize, but I encountered an error while processing your query.",
                "source_documents": [],
                "metadata": {"error": str(e)},
            }

    def convert_mail_to_json(self, mail) -> Dict[str, Any]:
        """Convert mail to JSON"""
        mail_json = {}
        mail_json["message_id"] = mail.message_id
        mail_json["subject"] = mail.subject
        mail_json["content"] = mail.content
        mail_json["thread_url"] = mail.thread_url
        mail_json["parent"] = mail.parent
        mail_json["children"] = mail.children
        mail_json["sender_address"] = mail.sender_address
        mail_json["from_field"] = mail.from_field
        mail_json["date"] = mail.date
        mail_json["to"] = mail.to
        mail_json["cc"] = mail.cc
        mail_json["reply_to"] = mail.reply_to
        mail_json["url"] = mail.url
        return mail_json
    
    
    def add_mail_data(self, mail_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add new documents to the RAG pipeline (both vector store and BM25) with duplicate checking
        
        Args:
            mail_data: List of mail messages to add
            
        Returns:
            Dictionary with statistics:
                - added: Number of messages added
                - skipped: Number of messages skipped (already exist)
                - failed: Number of messages that failed to process
                - failed_messages: List of message IDs that failed
        """
        self.logger.info("Processing %d mail documents for addition", len(mail_data))

        # Convert to Document objects
        failed_messages = []
        updated_count = 0
        added_count = 0
        for mail in mail_data:
            message_id = 'Unknown'
            try:
                if isinstance(mail, dict):
                    mail_json = mail
                    message_id = mail.get('message_id', 'Unknown')
                else:
                    mail_json = self.convert_mail_to_json(mail)
                    message_id = mail.message_id if hasattr(mail, 'message_id') else 'Unknown'
                
                if self.base_retriever.document_exists(mail_json['url']):
                    self.update_mail_data(mail_json)
                    updated_count += 1
                    continue
                
                doc = self.data_processor.process_mail_list([mail_json])
                # Split new documents using data processor
                chunked_docs = self.data_processor.chunk_documents(doc)
                self.base_retriever.add_documents(chunked_docs)
                added_count += 1
            except Exception as e:
                self.logger.exception("Error processing mail data: %s", e)
                failed_messages.append(message_id)

        # Clear cache after adding documents
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared after adding documents")

        # Calculate statistics
        total_processed = len(mail_data)
        
        result = {
            "added_count": added_count,
            "updated_count": updated_count,
            "failed_count": len(failed_messages),
            "failed_messages": failed_messages,
            "total_processed": total_processed
        }
        
        self.logger.info(
            f"Mail data processing complete: {added_count} added, {updated_count} updated (already exist), "
            f"{len(failed_messages)} failed out of {total_processed} total"
        )
        
        if failed_messages:
            self.logger.warning(f"Failed messages: {failed_messages}")
            
        return result

    def update_mail_data(self, message: Any) -> bool:
        """Update an existing document in both vector store and BM25
        
        Args:
            message: Dictionary containing mail message data or mail object
            
        Returns:
            True if update successful, False otherwise
        """
        # Get message_id properly based on type
        if isinstance(message, dict):
            message_id = message.get('message_id', 'Unknown')
            message_dict = message
        else:
            message_id = getattr(message, 'message_id', 'Unknown')
            message_dict = self.convert_mail_to_json(message)
        
        self.logger.info(f"Updating document {message_id}")

        try:
            # Create updated document from dict
            updated_doc = self.data_processor.process_mail_list([message_dict])
            # Chunk the updated document
            chunked_docs = self.data_processor.chunk_documents(updated_doc)
            # Update in hybrid retriever (use base retriever to bypass cache)
            self.base_retriever.update_documents(chunked_docs)
            return True
        except Exception as e:
            self.logger.exception("Error updating mail data: %s", e)
            return False

    def delete_document(self, doc_url: str):
        """Delete a document from both vector store and BM25"""
        self.logger.info(f"Deleting document {doc_url}")

        try:
            if self.base_retriever.document_exists(doc_url):
                doc_id = self.base_retriever.get_document_id(doc_url)
                self.base_retriever.delete_documents([doc_id])
                if self.cache:
                    self.cache.clear()
                    self.logger.info("Cache cleared after deleting document")
                return True
            else:
                self.logger.info(f"Document {doc_url} does not exist")
                return False    
            
        except Exception as e:
            self.logger.exception(f"Error deleting document {doc_id}: %s", e)
            raise

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        stats = {
            "total_documents": len(self.base_retriever.documents),
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "dense_top_k": self.base_retriever.dense_top_k,
            "sparse_top_k": self.base_retriever.sparse_top_k,
        }

        # Add cache stats if caching is enabled
        if self.cache:
            stats["cache"] = self.cache.get_stats()

        # Add telemetry stats if telemetry is enabled
        if self.telemetry:
            stats["telemetry"] = self.telemetry.get_aggregated_metrics()

        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {"error": "Caching is not enabled"}

    def get_telemetry_report(self) -> Dict[str, Any]:
        """Get telemetry report"""
        if self.telemetry:
            return self.telemetry.get_aggregated_metrics()
        return {"error": "Telemetry is not enabled"}

    def print_performance_report(self):
        """Print formatted performance report"""
        if self.telemetry:
            self.telemetry.print_report()
        else:
            self.logger.info("Telemetry is not enabled")

    def clear_cache(self):
        """Clear query cache"""
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared manually")
        else:
            self.logger.info("Caching is not enabled")

    def cleanup_cache(self) -> Dict[str, int]:
        """Manually trigger cache cleanup. Returns cleanup statistics."""
        if self.cache:
            expired = self.cache.cleanup_expired_files()
            size_limit = self.cache.enforce_disk_size_limit()
            self.logger.info(
                f"Cache cleanup: {expired} expired files, {size_limit} for size limit"
            )
            return {"expired": expired, "size_limit": size_limit}
        else:
            self.logger.info("Caching is not enabled")
            return {"expired": 0, "size_limit": 0}

    def get_cache_size_info(self) -> Dict[str, Any]:
        """Get cache size information"""
        if self.cache:
            return self.cache.get_cache_size_info()
        return {"error": "Caching is not enabled"}

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        responses = []

        for question in tqdm(questions, desc="Processing queries"):
            response = self.query(question)
            responses.append(response)

        return responses


# Example usage and testing
def create_langchain_rag_pipeline(config: LangChainConfig = None, **kwargs):
    """Create and return a configured LangChain RAG pipeline"""
    if config is None:
        config = (
            LangChainConfig.from_env() if not kwargs else LangChainConfig(**kwargs)
        )

    return LangChainRAGPipeline(config=config)


if __name__ == "__main__":
    # Example usage
    pipeline = create_langchain_rag_pipeline(
        mail_data_dir="data/processed/message_by_thread",
        doc_data_dir="data/source_data/processed/en",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chroma_persist_dir="./langchain_rag/chroma_db",
        force_reindex=False,
    )

    # Test retrieval
    test_question = "How does Boost.Asio handle asynchronous operations?"
    docs = pipeline.retrieve(test_question, fetch_k=5)

    print(f"Question: {test_question}")
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. Score: {doc.metadata.get('final_score', 0):.3f}")
        print(f"   Type: {doc.metadata.get('type', 'unknown')}")
        print(f"   Content: {doc.page_content[:200]}...")
