"""
Cross-encoder for reranking retrieval results.
"""

from typing import List, Optional, Any
from loguru import logger
from sentence_transformers import CrossEncoder
from utils.config import get_config


class CrossEncoderReranker:
    """Cross-encoder for reranking retrieval results."""
    
    def __init__(self, model_name: Optional[str] = None, shared_model=None, language: str = None):
        self.logger = logger.bind(name="CrossEncoderReranker")
        
        # Use shared model if provided
        if language is not None:
            self.language = language
            
        if shared_model:
            self.model = shared_model
            self.model_name = "shared_model"
            self.logger.info("✅ Using shared reranker model")
        else:
            if model_name is None:
                model_name = get_config("rag.reranker.model_name", "Alibaba-NLP/gte-multilingual-reranker-base")
            
            self.model_name = model_name
            self.model = None
            
            self.load_model()
        
        
        self.top_k_before_rerank = get_config("rag.reranker.top_k_before_rerank", 20)
        self.top_k_after_rerank = get_config("rag.reranker.top_k_after_rerank", 5)
    
    def load_model(self):
        """Load the cross-encoder model."""
        if self.model is None:
            try:
                self.logger.info(f"Loading cross-encoder model: {self.model_name}")
                self.model = CrossEncoder(self.model_name, trust_remote_code=True)
                self.logger.info(f"Cross-encoder model loaded: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Error loading cross-encoder model: {e}")
                raise
    
    def rerank(self, query: str, results: Any)-> Any:
        """Rerank results using cross-encoder."""
        if not results:
            return results
        
        if self.model is None:
            self.load_model()
        
        # Prepare query-document pairs
        pairs = [(query, result.text) for result in results]
        
        # Get relevance scores
        relevance_scores = self.model.predict(pairs)
        
        # Update scores and sort
        for result, score in zip(results, relevance_scores):
            result.score = float(score)
        
        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:self.top_k_after_rerank]
    def reset_model(self, model_name: str):
        self.model_name = model_name
        
        self.load_model()