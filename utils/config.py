"""
Configuration management utilities for VectorDataBuild project.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    text: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str  # 'embedding', 'bm25', 'graph', 'hybrid', 'hierarchical'
    source_type: str # 'code', 'text', 'email'
    source_file: str


# Global config cache
_config_cache: Optional[Dict[str, Any]] = None

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return _config_cache
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        key: Configuration key (e.g., 'rag.embedding_model')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = load_config()
    
    if not config:
        return default
    
    keys = key.split('.')
    value = config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def update_config(key: str, value: Any) -> bool:
    """
    Update configuration value.
    
    Args:
        key: Configuration key (e.g., 'rag.embedding_model')
        value: New value
        
    Returns:
        True if successful, False otherwise
    """
    global _config_cache
    
    if _config_cache is None:
        _config_cache = load_config()
    
    keys = key.split('.')
    config = _config_cache
    
    try:
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        return True
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return False

def save_config(config_path: str = "src/config/config.yaml") -> bool:
    """
    Save configuration to YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if successful, False otherwise
    """
    global _config_cache
    
    if _config_cache is None:
        return False
    
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(_config_cache, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def get_model_nick_name(model_name: str) -> str:
    """Get the model nick name."""
    for nick_name in get_config("rag.embedding.embedding_types_list", ["gemma", "minilm", "nomic", "jina", "baai"]):
        if model_name == get_config(f"rag.embedding.{nick_name}.model_name"):
            return nick_name
    return get_config("rag.embedding.default_embedding_type", "gemma")
