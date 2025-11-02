"""
LangChain RAG Pipeline Configuration
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class LangChainConfig:
    """Configuration class for LangChain RAG pipeline"""

    # Data settings
    mail_data_dir: str = "data/processed/message_by_thread"
    doc_data_dir: str = "data/source_data/processed/en"
    # chunk_size: int = 5000  # for embeddinggemma-300m
    # chunk_overlap: int = 300  # for embeddinggemma-300m
    chunk_size: int = 1024   # for sentence-transformers/all-MiniLM-L6-v2
    chunk_overlap: int = 100  # for sentence-transformers/all-MiniLM-L6-v2

    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "ollama_llama2_7b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # Vector store settings
    retriever_type: str = "chroma"  # Options: "chroma", "faiss", "duckdb"
    chroma_persist_dir: str = "langchain_rag/chroma_db"
    chroma_collection_name: str = "boost_docs"
    faiss_persist_dir: str = "langchain_rag/faiss_db"
    duckdb_path: str = "langchain_rag/duckdb/duckdb_retriever.db"

    # Retrieval settings
    dense_top_k: int = 100
    sparse_top_k: int = 100
    final_top_k: int = 10
    half_life: int = 1500
    # DuckDB-specific settings
    vector_batch_size: int = 1000  # Batch size for loading/storing vectors

    # TF-IDF settings
    max_features: int = 10000
    ngram_range: tuple = (1, 2)

    # Performance settings
    use_cuda: bool = True
    show_progress: bool = True

    # Persistence settings
    force_reindex: bool = False

    # Cache settings
    enable_cache: bool = True
    cache_dir: str = "langchain_rag/query_cache"
    cache_ttl_seconds: int = 3600  # 1 hour default TTL
    cache_max_memory_entries: int = 1000  # Max entries in memory cache
    cache_max_disk_size_mb: int = 500  # Max disk cache size in MB
    cache_auto_cleanup_interval: int = 100  # Cleanup every N operations

    # Telemetry settings
    enable_telemetry: bool = True
    telemetry_log_file: str = "logs/monitoring_metrics.json"
    
    # Logging settings
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "LangChainConfig":
        """Create config from environment variables"""
        return cls(
            mail_data_dir=os.getenv("LANGCHAIN_MAIL_DATA_DIR", "data/processed/message_by_thread"),
            doc_data_dir=os.getenv("LANGCHAIN_DOC_DATA_DIR", "data/source_data/processed/en"),
            chunk_size=int(os.getenv("LANGCHAIN_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("LANGCHAIN_CHUNK_OVERLAP", "50")),
            embedding_model=os.getenv(
                "LANGCHAIN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_model=os.getenv("LANGCHAIN_LLM_MODEL", "ollama_llama2_7b"),
            llm_temperature=float(os.getenv("LANGCHAIN_LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LANGCHAIN_LLM_MAX_TOKENS", "1024")),
            retriever_type=os.getenv("LANGCHAIN_RETRIEVER_TYPE", "chroma"),
            chroma_persist_dir=os.getenv(
                "LANGCHAIN_CHROMA_DIR", "./chroma_db_langchain"
            ),
            chroma_collection_name=os.getenv(
                "LANGCHAIN_CHROMA_COLLECTION", "boost_docs"
            ),
            duckdb_path=os.getenv("LANGCHAIN_DUCKDB_PATH", "langchain_rag/duckdb_retriever.db"),
            dense_top_k=int(os.getenv("LANGCHAIN_DENSE_TOP_K", "10")),
            sparse_top_k=int(os.getenv("LANGCHAIN_SPARSE_TOP_K", "10")),
            final_top_k=int(os.getenv("LANGCHAIN_FINAL_TOP_K", "15")),
            vector_batch_size=int(os.getenv("LANGCHAIN_VECTOR_BATCH_SIZE", "1000")),
            max_features=int(os.getenv("LANGCHAIN_MAX_FEATURES", "10000")),
            use_cuda=os.getenv("LANGCHAIN_USE_CUDA", "true").lower() == "true",
            show_progress=os.getenv("LANGCHAIN_SHOW_PROGRESS", "true").lower()
            == "true",
            force_reindex=os.getenv("LANGCHAIN_FORCE_REINDEX", "false").lower()
            == "true",
            enable_cache=os.getenv("LANGCHAIN_ENABLE_CACHE", "true").lower()
            == "true",
            cache_dir=os.getenv("LANGCHAIN_CACHE_DIR", "data/cache"),
            cache_ttl_seconds=int(os.getenv("LANGCHAIN_CACHE_TTL", "3600")),
            cache_max_memory_entries=int(os.getenv("LANGCHAIN_CACHE_MAX_MEMORY", "1000")),
            cache_max_disk_size_mb=int(os.getenv("LANGCHAIN_CACHE_MAX_DISK_MB", "500")),
            cache_auto_cleanup_interval=int(os.getenv("LANGCHAIN_CACHE_CLEANUP_INTERVAL", "100")),
            enable_telemetry=os.getenv("LANGCHAIN_ENABLE_TELEMETRY", "true").lower()
            == "true",
            telemetry_log_file=os.getenv(
                "LANGCHAIN_TELEMETRY_LOG", "logs/monitoring_metrics.json"
            ),
            log_level=os.getenv("LANGCHAIN_LOG_LEVEL", "INFO"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "mail_data_dir": self.mail_data_dir,
            "doc_data_dir": self.doc_data_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "retriever_type": self.retriever_type,
            "chroma_persist_dir": self.chroma_persist_dir,
            "chroma_collection_name": self.chroma_collection_name,
            "duckdb_path": self.duckdb_path,
            "dense_top_k": self.dense_top_k,
            "sparse_top_k": self.sparse_top_k,
            "final_top_k": self.final_top_k,
            "vector_batch_size": self.vector_batch_size,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "use_cuda": self.use_cuda,
            "show_progress": self.show_progress,
            "force_reindex": self.force_reindex,
            "enable_cache": self.enable_cache,
            "cache_dir": self.cache_dir,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_max_memory_entries": self.cache_max_memory_entries,
            "cache_max_disk_size_mb": self.cache_max_disk_size_mb,
            "cache_auto_cleanup_interval": self.cache_auto_cleanup_interval,
            "enable_telemetry": self.enable_telemetry,
            "telemetry_log_file": self.telemetry_log_file,
            "log_level": self.log_level,
        }

    def validate(self) -> bool:
        """Validate configuration"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if not os.path.exists(self.mail_data_dir):
            raise ValueError(f"mail_data_dir does not exist: {self.mail_data_dir}")
        if not os.path.exists(self.doc_data_dir):
            raise ValueError(f"doc_data_dir does not exist: {self.doc_data_dir}")
        if self.retriever_type not in ["chroma", "faiss", "duckdb"]:
            raise ValueError(f"retriever_type must be 'chroma', 'faiss', or 'duckdb', got: {self.retriever_type}")
        if self.vector_batch_size <= 0:
            raise ValueError("vector_batch_size must be positive")
        return True


# Default configuration instance
DEFAULT_CONFIG = LangChainConfig()

