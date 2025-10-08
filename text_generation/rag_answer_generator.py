"""
RAG Answer Generator Module
Generates answers based on RAG retrieval results using text generation models.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from text_generation.text_generation_model import TextGenerationModel
from utils.config import get_config


class RAGAnswerGenerator:
    """
    Generates answers based on RAG retrieval results.
    
    This module takes retrieval results from the RAG system and uses
    a text generation model to create comprehensive answers.
    """
    
    def __init__(self, text_generation_model: TextGenerationModel = None):
        """
        Initialize the RAG answer generator.
        
        Args:
            text_generation_model: Text generation model to use for answer generation
        """
        self.logger = logger.bind(name="RAGAnswerGenerator")
        self.text_generation_model = text_generation_model
        self.is_ready = text_generation_model is not None and text_generation_model.is_model_ready()
        
        if self.is_ready:
            self.logger.info("✅ RAG Answer Generator initialized successfully")
        else:
            self.logger.warning("⚠️ RAG Answer Generator initialized without text generation model")
    
    def generate_answer(self, 
                       question: str, 
                       retrieval_results: List[Dict[str, Any]], 
                       max_context_length: int = 4000,
                       include_sources: bool = True) -> Dict[str, Any]:
        """
        Generate an answer based on retrieval results.
        
        Args:
            question: User's question
            retrieval_results: List of retrieval results from RAG system
            max_context_length: Maximum length of context to include
            include_sources: Whether to include source information in the answer
            
        Returns:
            Dictionary containing:
                - answer: Generated answer text
                - sources: List of sources used
                - context_used: Number of results used for context
                - generation_metadata: Additional metadata about generation
        """
        try:
            if not self.is_ready:
                return {
                    "answer": "I'm sorry, but the text generation model is not available. Please try again later.",
                    "sources": [],
                    "context_used": 0,
                    "generation_metadata": {"error": "Text generation model not ready"}
                }
            
            if not retrieval_results:
                return {
                    "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your question or ask about a different topic.",
                    "sources": [],
                    "context_used": 0,
                    "generation_metadata": {"warning": "No retrieval results provided"}
                }
            
            # Build context from retrieval results
            context, sources_used = self._build_context_from_results(
                retrieval_results, 
                max_context_length
            )
            
            # Generate answer using the text generation model
            answer = self.text_generation_model.generate_rag_answer(
                question=question,
                retrieval_results=retrieval_results[:len(sources_used)]
            )
            
            # Prepare sources information
            sources = []
            if include_sources:
                for result in sources_used:
                    sources.append({
                        "source_file": result.get('source_file', 'Unknown'),
                        "score": result.get('score', 0),
                        "retrieval_method": result.get('retrieval_method', 'unknown'),
                        "text_preview": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', '')
                    })
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": len(sources_used),
                "generation_metadata": {
                    "model_used": self.text_generation_model.model_name,
                    "context_length": len(context),
                    "total_results": len(retrieval_results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating RAG answer: {e}")
            return {
                "answer": f"I encountered an error while generating an answer: {str(e)}",
                "sources": [],
                "context_used": 0,
                "generation_metadata": {"error": str(e)}
            }
    
    def _build_context_from_results(self, 
                                  retrieval_results: List[Dict[str, Any]], 
                                  max_length: int) -> tuple[str, List[Dict[str, Any]]]:
        """
        Build context string from retrieval results, respecting max_length.
        
        Args:
            retrieval_results: List of retrieval results
            max_length: Maximum length of context
            
        Returns:
            Tuple of (context_string, sources_used)
        """
        context_parts = []
        sources_used = []
        current_length = 0
        
        for i, result in enumerate(retrieval_results):
            source_file = result.get('source_file', 'Unknown source')
            text = result.get('text', '')
            score = result.get('score', 0)
            method = result.get('retrieval_method', 'unknown')
            
            # Estimate length of this context part
            context_part = f"[Source {i+1}] {source_file} (Score: {score:.3f}, Method: {method})\n{text}\n"
            part_length = len(context_part)
            
            # Check if adding this part would exceed max_length
            if current_length + part_length > max_length and current_length > 0:
                break
            
            context_parts.append(context_part)
            sources_used.append(result)
            current_length += part_length
        
        return "\n".join(context_parts), sources_used
    
    def is_ready(self) -> bool:
        """
        Check if the answer generator is ready to use.
        
        Returns:
            True if ready, False otherwise
        """
        return self.is_ready and self.text_generation_model is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the answer generator.
        
        Returns:
            Dictionary containing generator statistics
        """
        return {
            "is_ready": self.is_ready,
            "text_generation_model": self.text_generation_model.model_name if self.text_generation_model else None,
            "model_ready": self.text_generation_model.is_model_ready() if self.text_generation_model else False
        }
