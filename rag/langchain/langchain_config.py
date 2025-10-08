"""
Configuration management for LangChain RAG Pipeline.
Provides easy configuration loading and validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import get_config


@dataclass
class LangChainRAGConfig:
    """Configuration for LangChain RAG Pipeline."""
    
    # Data paths
    raw_data_path: str = "data/processed/raw"
    chunked_data_path: str = "data/processed/chunked"
    embeddings_path: str = "data/processed/embeddings"
    graph_cache_path: str = "data/processed/graph_cache"
    hierarchical_cache_path: str = "data/processed/hierarchical_cache"
    
    # Model configurations
    embedding_model: str = "gemma"
    llm_type: str = "ollama"
    llm_model: str = "gemma3:1b"
    language: str = "en"
    
    # Retrieval configuration
    top_k: int = 10
    similarity_threshold: float = 0.7
    use_reranker: bool = True
    
    # Neo4j configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # Chunking configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic_similarity_threshold: float = 0.7
    
    # Retrieval weights
    dense_weight: float = 0.5
    sparse_weight: float = 0.2
    graph_weight: float = 0.2
    hierarchical_weight: float = 0.1


class LangChainConfigManager:
    """Manager for LangChain RAG Pipeline configuration."""
    
    def __init__(self):
        self.logger = logger.bind(name="LangChainConfigManager")
    
    def load_from_yaml(self, config_path: str = None) -> LangChainRAGConfig:
        """Load configuration from YAML file."""
        try:
            # Get default values from main config
            embedding_model = get_config("rag.embedding.default_embedding_type", "gemma")
            llm_type = get_config("rag.llm.default_llm_type", "ollama")
            language = get_config("language.default_language", "en")
            
            # Get Neo4j configuration
            neo4j_config = get_config("rag.graph.neo4j", {})
            neo4j_uri = neo4j_config.get("uri", "bolt://localhost:7687")
            neo4j_username = neo4j_config.get("username", "neo4j")
            neo4j_password = neo4j_config.get("password", "password")
            neo4j_database = neo4j_config.get("database", "neo4j")
            
            # Get retrieval configuration
            top_k = get_config("rag.retrieval.top_k", 10)
            similarity_threshold = get_config("rag.retrieval.similarity_threshold", 0.7)
            
            # Get chunking configuration
            chunk_size = get_config("rag.chunking.max_chunk_size", 1000)
            chunk_overlap = get_config("rag.chunking.chunk_overlap", 200)
            semantic_threshold = get_config("rag.chunking.semantic_similarity_threshold", 0.7)
            
            # Get retrieval weights
            retrieval_weights = get_config("rag.retrieval.retrieval_default_weight", {
                "vector search": 0.5,
                "bm25 search": 0.2,
                "graph search": 0.15,
                "hierarchical search": 0.15
            })
            
            config = LangChainRAGConfig(
                embedding_model=embedding_model,
                llm_type=llm_type,
                language=language,
                neo4j_uri=neo4j_uri,
                neo4j_username=neo4j_username,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                semantic_similarity_threshold=semantic_threshold,
                dense_weight=retrieval_weights.get("vector search", 0.5),
                sparse_weight=retrieval_weights.get("bm25 search", 0.2),
                graph_weight=retrieval_weights.get("graph search", 0.15),
                hierarchical_weight=retrieval_weights.get("hierarchical search", 0.15)
            )
            
            self.logger.info("✅ Configuration loaded from YAML")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            # Return default configuration
            return LangChainRAGConfig()
    
    def validate_config(self, config: LangChainRAGConfig) -> bool:
        """Validate configuration."""
        try:
            # Check required paths
            required_paths = [
                config.raw_data_path,
                config.chunked_data_path,
                config.embeddings_path
            ]
            
            for path in required_paths:
                if not os.path.exists(path):
                    self.logger.warning(f"⚠️ Path does not exist: {path}")
            
            # Check model configurations
            if not config.embedding_model:
                self.logger.error("❌ Embedding model not specified")
                return False
            
            if not config.llm_type:
                self.logger.error("❌ LLM type not specified")
                return False
            
            # Check retrieval weights sum to 1.0
            total_weight = (config.dense_weight + config.sparse_weight + 
                          config.graph_weight + config.hierarchical_weight)
            
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                self.logger.warning(f"⚠️ Retrieval weights sum to {total_weight}, not 1.0")
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def save_config(self, config: LangChainRAGConfig, filepath: str):
        """Save configuration to file."""
        try:
            config_dict = asdict(config)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                import json
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Configuration saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")
    
    def load_config(self, filepath: str) -> Optional[LangChainRAGConfig]:
        """Load configuration from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import json
                config_dict = json.load(f)
            
            config = LangChainRAGConfig(**config_dict)
            self.logger.info(f"✅ Configuration loaded from: {filepath}")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            return None


def get_langchain_config() -> LangChainRAGConfig:
    """Get LangChain RAG configuration."""
    manager = LangChainConfigManager()
    config = manager.load_from_yaml()
    
    if manager.validate_config(config):
        return config
    else:
        logger.warning("⚠️ Configuration validation failed, using defaults")
        return LangChainRAGConfig()


if __name__ == "__main__":
    # Test configuration loading
    config = get_langchain_config()
    print(f"Configuration: {config}")
    
    # Test validation
    manager = LangChainConfigManager()
    is_valid = manager.validate_config(config)
    print(f"Configuration valid: {is_valid}")
