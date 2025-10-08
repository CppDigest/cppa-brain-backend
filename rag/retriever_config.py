"""
Clean configuration system for the reorganized retrieval architecture.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from rag.new_hybrid_retriever import HybridRetriever
from utils.config import get_config


class RetrieverType(Enum):
    """Types of retrievers available."""
    DOCUMENT = "document"
    MESSAGE = "message"
    HYBRID = "hybrid"

@dataclass
class DocumentRetrieverConfig:
    """Configuration for document retriever."""
    data_path: Optional[Path] = None
    language: str = "en"
    use_embedder: bool = True
    use_graph: bool = True
    use_bm25: bool = True
    use_reranker: bool = True
    max_results: int = 10


@dataclass
class MessageRetrieverConfig:
    """Configuration for message retriever."""
    data_path: Optional[Path] = None
    language: str = "en"
    use_reranker: bool = True
    max_results: int = 10
    cache_graph: bool = True
    graph_cache_path: Optional[Path] = None


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retriever."""
    language: str = "en"
    document_config: Optional[DocumentRetrieverConfig] = None
    message_config: Optional[MessageRetrieverConfig] = None
    default_scope: str = "both"  # "documents_only", "messages_only", "both"
    document_weight: float = 0.6
    message_weight: float = 0.4


class RetrieverConfigManager:
    """Manages configuration for all retrievers."""
    
    def __init__(self):
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from config files."""
        return {
            "language": get_config("language.default_language", "en"),
            "document_data_path": get_config("data.processed_data.chunked_data_path", "data/processed/chunked"),
            "message_data_path": get_config("data.processed_data.message_by_thread_path", "data/processed/message_by_thread"),
            "graph_cache_path": get_config("data.processed_data.graph_save_path", "data/processed/graph_cache"),
            "max_results": get_config("rag.retrieval.top_k", 10),
            "use_reranker": get_config("rag.retrieval.use_reranker", True),
            "document_weight": get_config("rag.hybrid_retrieval.document_weight", 0.6),
            "message_weight": get_config("rag.hybrid_retrieval.message_weight", 0.4)
        }
    
    def get_document_config(self, **overrides) -> DocumentRetrieverConfig:
        """Get document retriever configuration."""
        config = DocumentRetrieverConfig(
            data_path=Path(self.config["document_data_path"]) / self.config["language"],
            language=self.config["language"],
            use_embedder=True,
            use_graph=True,
            use_bm25=True,
            use_reranker=self.config["use_reranker"],
            max_results=self.config["max_results"]
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def get_message_config(self, **overrides) -> MessageRetrieverConfig:
        """Get message retriever configuration."""
        config = MessageRetrieverConfig(
            data_path=Path(self.config["message_data_path"]),
            language=self.config["language"],
            use_reranker=self.config["use_reranker"],
            max_results=self.config["max_results"],
            cache_graph=True,
            graph_cache_path=Path(self.config["graph_cache_path"])
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def get_hybrid_config(self, **overrides) -> HybridRetrieverConfig:
        """Get hybrid retriever configuration."""
        config = HybridRetrieverConfig(
            language=self.config["language"],
            document_config=self.get_document_config(),
            message_config=self.get_message_config(),
            default_scope="both",
            document_weight=self.config["document_weight"],
            message_weight=self.config["message_weight"]
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def update_config(self, **updates):
        """Update configuration values."""
        self.config.update(updates)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        return self.config.get(key, default)


class RetrieverFactory:
    """Factory for creating retrievers with proper configuration."""
    
    def __init__(self, config_manager: Optional[RetrieverConfigManager] = None):
        self.config_manager = config_manager or RetrieverConfigManager()
    
    def create_document_retriever(self, embedder=None, graph_search_engine=None, 
                                 sparse_search_engine=None, reranker=None, **config_overrides):
        """Create a document retriever with configuration."""
        from rag.unstructured_retriever import DocumentRetriever
        
        config = self.config_manager.get_document_config(**config_overrides)
        
        return DocumentRetriever(
            language=config.language,
            embedder=embedder,
            graph_search_engine=graph_search_engine,
            sparse_search_engine=sparse_search_engine,
            reranker=reranker
        )
    
    def create_message_retriever(self, mail_hierarchy_graph=None, reranker=None, **config_overrides):
        """Create a message retriever with configuration."""
        from rag.structured_retriever import MessageRetriever
        
        config = self.config_manager.get_message_config(**config_overrides)
        
        return MessageRetriever(
            language=config.language,
            mail_hierarchy_graph=mail_hierarchy_graph,
            reranker=reranker
        )
    
    def create_hybrid_retriever(self, dense_search_engine=None, graph_search_engine=None, sparse_search_engine=None,
                               mail_hierarchy_graph=None, reranker=None, **config_overrides):
        """Create a hybrid retriever with configuration."""
        
        config = self.config_manager.get_hybrid_config(**config_overrides)
        
        return HybridRetriever(
            language=config.language,
            dense_search_engine=dense_search_engine,
            graph_search_engine=graph_search_engine,
            sparse_search_engine=sparse_search_engine,
            mail_hierarchy_graph=mail_hierarchy_graph,
            reranker=reranker
        )

# Global configuration manager instance
config_manager = RetrieverConfigManager()
retriever_factory = RetrieverFactory(config_manager)
