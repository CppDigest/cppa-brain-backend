"""
Text Generation Model - Parent class for all chatbot implementations.
Provides unified interface for different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from loguru import logger
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_config


class TextGenerationModel(ABC):
    """Abstract base class for all text generation models."""
    
    def __init__(self, model_name: str = None, config: Dict[str, Any] = None):
        """
        Initialize text generation model.
        
        Args:
            model_name: Name of the model to use
            config: Configuration dictionary for the model
        """
        self.logger = logger.bind(name="TextGenerationModel")
        self.model_name = model_name
        self.config = config or {}
        self.is_initialized = False
        self.is_ready = False
        
        self.logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the text generation model.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_response(self, user_input: str, **kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            user_input: Input text from user
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated response text
        """
        pass
    
    def generate_rag_answer(self, question: str, retrieval_results: List[Dict[str, Any]], **kwargs) -> str:
        """
        Generate an answer based on RAG retrieval results.
        
        Args:
            question: User's question
            retrieval_results: List of retrieval results with text, score, source_file, etc.
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated answer based on retrieval results
        """
        if not retrieval_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Create context from retrieval results
        context = self._build_context_from_results(retrieval_results)
        
        # Create prompt for RAG-based answer generation
        prompt = self._create_rag_prompt(question, context)
        
        # Generate response using the context
        return self.generate_response(prompt, **kwargs)
    
    def _build_context_from_results(self, retrieval_results: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            retrieval_results: List of retrieval results
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(retrieval_results, 1):
            source_file = result.get('source_file', 'Unknown source')
            text = result.get('text', '')
            score = result.get('score', 0)
            method = result.get('retrieval_method', 'unknown')
            
            # Truncate very long text
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            context_part = f"[Source {i}] {source_file} (Score: {score:.3f}, Method: {method})\n{text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for RAG-based answer generation.
        
        Args:
            question: User's question
            context: Context from retrieval results
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""Based on the following context from Boost C++ library documentation, please answer the user's question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please say so and provide what information is available. Make sure to reference specific sources when possible.

Answer:"""
        
        return prompt
    
    @abstractmethod
    def is_model_ready(self) -> bool:
        """
        Check if the model is ready for use.
        
        Returns:
            True if model is ready, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "is_initialized": self.is_initialized,
            "is_ready": self.is_ready,
            "config": self.config
        }
    
    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Update model configuration.
        
        Args:
            config: New configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config.update(config)
            self.logger.info(f"Configuration updated for {self.__class__.__name__}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    def reset_model(self, model_name: str = None) -> bool:
        """
        Reset the model with a new model name.
        
        Args:
            model_name: New model name to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name:
                self.model_name = model_name
                self.logger.info(f"Model name updated to: {model_name}")
            
            self.is_initialized = False
            self.is_ready = False
            
            return self.initialize()
        except Exception as e:
            self.logger.error(f"Error resetting model: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get model capabilities and features.
        
        Returns:
            Dictionary containing capability information
        """
        return {
            "supports_streaming": False,
            "supports_function_calling": False,
            "supports_vision": False,
            "supports_audio": False,
            "max_tokens": 4096,
            "supports_system_prompt": True,
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_frequency_penalty": False,
            "supports_presence_penalty": False
        }
    
    def validate_input(self, user_input: str) -> bool:
        """
        Validate input text.
        
        Args:
            user_input: Input text to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not user_input or not isinstance(user_input, str):
            self.logger.warning("Invalid input: empty or non-string input")
            return False
        
        if len(user_input.strip()) == 0:
            self.logger.warning("Invalid input: empty string after stripping")
            return False
        
        return True
    
    def preprocess_input(self, user_input: str) -> str:
        """
        Preprocess input text before generation.
        
        Args:
            user_input: Raw input text
            
        Returns:
            Preprocessed input text
        """
        # Basic preprocessing - can be overridden by subclasses
        return user_input.strip()
    
    def postprocess_output(self, output: str) -> str:
        """
        Postprocess output text after generation.
        
        Args:
            output: Raw output text
            
        Returns:
            Postprocessed output text
        """
        # Basic postprocessing - can be overridden by subclasses
        return output.strip()
    
    def generate_with_preprocessing(self, user_input: str, **kwargs) -> str:
        """
        Generate response with input preprocessing and output postprocessing.
        
        Args:
            user_input: Input text from user
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated and postprocessed response text
        """
        if not self.validate_input(user_input):
            return "Invalid input provided."
        
        if not self.is_model_ready():
            self.logger.warning("Model is not ready, attempting to initialize...")
            if not self.initialize():
                return "Model is not available."
        
        try:
            # Preprocess input
            processed_input = self.preprocess_input(user_input)
            
            # Generate response
            response = self.generate_response(processed_input, **kwargs)
            
            # Postprocess output
            final_response = self.postprocess_output(response)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error in text generation: {e}")
            return f"Error generating response: {str(e)}"
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model={self.model_name}, ready={self.is_ready})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', initialized={self.is_initialized}, ready={self.is_ready})"


class TextGenerationModelFactory:
    """Factory class for creating text generation models."""
    
    @staticmethod
    def create_model(model_group: str, model_name: str = None, config: Dict[str, Any] = None) -> TextGenerationModel:
        """
        Create a text generation model instance.
        
        Args:
            model_group: Group of model ('ollama', 'openai', 'gemini', 'huggingface')
            model_name: Name of the model to use
            config: Configuration dictionary
            
        Returns:
            TextGenerationModel instance
        """
        try:
            if model_group.lower() == 'ollama':
                from text_generation.ollama_chatbot import OllamaChatbot
                return OllamaChatbot(model_name=model_name, config=config)
            
            elif model_group.lower() == 'openai':
                from text_generation.openai_chatbot import OpenAIChatbot
                return OpenAIChatbot(model_name=model_name, config=config)
            
            elif model_group.lower() == 'gemini':
                from text_generation.gemini_chatbot import GeminiChatbot
                return GeminiChatbot(model_name=model_name, config=config)
            
            elif model_group.lower() == 'huggingface':
                from text_generation.huggingface_chatbot import HuggingFaceChatbot
                return HuggingFaceChatbot(model_name=model_name, config=config)
            
            else:
                raise ValueError(f"Unsupported model type: {model_group}")
                
        except ImportError as e:
            logger.error(f"Failed to import {model_group} chatbot: {e}")
            raise ValueError(f"Model type {model_group} is not available")
        except Exception as e:
            logger.error(f"Error creating {model_group} model: {e}")
            raise
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of available model type names
        """
        return ['ollama', 'openai', 'gemini', 'huggingface']
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model type.
        
        Args:
            model_type: Type of model to get info for
            
        Returns:
            Dictionary containing model type information
        """
        model_info = {
            'ollama': {
                'name': 'Ollama',
                'description': 'Local Ollama models',
                'supports_local': True,
                'supports_remote': False,
                'default_model': 'gemma3:1b'
            },
            'openai': {
                'name': 'OpenAI',
                'description': 'OpenAI API models',
                'supports_local': False,
                'supports_remote': True,
                'default_model': 'gpt-3.5-turbo'
            },
            'gemini': {
                'name': 'Google Gemini',
                'description': 'Google Gemini API models',
                'supports_local': False,
                'supports_remote': True,
                'default_model': 'gemini-pro'
            },
            'huggingface': {
                'name': 'Hugging Face',
                'description': 'Hugging Face Transformers models',
                'supports_local': True,
                'supports_remote': True,
                'default_model': 'microsoft/DialoGPT-medium'
            }
        }
        
        return model_info.get(model_type.lower(), {
            'name': 'Unknown',
            'description': 'Unknown model type',
            'supports_local': False,
            'supports_remote': False,
            'default_model': None
        })
