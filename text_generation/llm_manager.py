"""
Shared LLM Manager for RAG system.
Manages a single LLM instance that can be shared across all RAG components.
"""

import os
import sys
from typing import Optional, Dict, Any, List
from loguru import logger
import torch
from huggingface_hub import InferenceApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_config


class LLMManager:
    """Shared LLM manager that initializes and manages a single LLM instance."""
    
    def __init__(self):
        self.logger = logger.bind(name="LLMManager")
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_initialized = False
        
        # Configuration
        self.model_name = get_config("rag.llm.model_name", "gpt2")
        self.model_path = get_config("rag.llm.model_path", None)  # Local path to downloaded model
        self.temperature = get_config("rag.llm.temperature", 0.7)
        self.max_tokens = get_config("rag.llm.max_tokens", 512)
        self.device_name = get_config("rag.llm.device", "auto")
        self.model_max_length = get_config("rag.llm.model_max_length", 1024)  # GPT-2 default
        self.api = InferenceApi(repo_id="EleutherAI/gpt-j-6B")

        self.initialize()

        self.logger.info("LLMManager initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the LLM model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing LLM: {self.model_name}")
            
            # Determine device
            if self.device_name == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_name)
            
            self.logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            if self.model_path and os.path.exists(self.model_path):
                self.logger.info(f"Loading tokenizer from local path: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                self.logger.info(f"Loading tokenizer from HuggingFace: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if self.model_path and os.path.exists(self.model_path):
                self.logger.info(f"Loading model from local path: {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            else:
                self.logger.info(f"Loading model from HuggingFace: {self.model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.is_initialized = True
            self.logger.info("LLM initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            self.is_initialized = False
            return False
    
    def generate_text(self, prompt: str, max_length: Optional[int] = None, 
                     temperature: Optional[float] = None, 
                     do_sample: bool = True) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        if not self.is_initialized:
            self.logger.error("LLM not initialized")
            return "LLM not initialized"
        
        try:
            # Use provided parameters or defaults
            max_len = max_length or self.max_tokens
            temp = temperature or self.temperature
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Check if input is too long and ensure total length doesn't exceed model limits
            max_input_length = self.model_max_length - max_len - 10  # Leave room for generation and safety
            
            if inputs.shape[1] > max_input_length:
                self.logger.warning(f"Input too long ({inputs.shape[1]} tokens), truncating to {max_input_length}")
                inputs = inputs[:, :max_input_length]
            
            # Generate
            with torch.no_grad():
                # Use max_length instead of min_new_tokens/max_new_tokens to avoid position embedding issues
                total_max_length = inputs.shape[1] + max_len
                
                # Ensure we don't exceed model's maximum context length
                if total_max_length > self.model_max_length:
                    total_max_length = self.model_max_length
                    self.logger.warning(f"Total length would exceed model limit, capping at {self.model_max_length}")
                
                # outputs = self.api(inputs = prompt, raw_response=True)
                outputs = self.model.generate(
                    inputs,
                    max_length=total_max_length,
                    temperature=temp,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # Decode output
            if outputs.numel() == 0:
                self.logger.warning("Empty output from model generation")
                return "No response generated"
            
            if outputs.shape[0] == 0:
                self.logger.warning("No sequences generated")
                return "No response generated"
            
            # Use the first generated sequence
            generated_sequence = outputs[0]
            generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            
            # Remove the original prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Ensure we have some content
            if not generated_text.strip():
                self.logger.warning("Generated text is empty after prompt removal")
                return "No meaningful response generated"
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"
    
    def generate_response(self, user_input: str, 
                              max_length: Optional[int] = 500,
                              temperature: Optional[float] = 0.7) -> str:
        """
        Generate a chat response from a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        messages = [{"role": "user", "content": user_input}]
        if not self.is_initialized:
            self.logger.error("LLM not initialized")
            return "LLM not initialized"
        
        try:
            # Convert messages to a single user_input
            user_input = self._format_messages_as_prompt(messages)
            
            # Generate response
            response = self.generate_text(
                prompt=user_input,
                max_length=max_length,
                temperature=temperature
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}")
            return f"Error generating chat response: {str(e)}"
    
    def _format_messages_as_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages as a single prompt for the model."""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        # Add prompt for the assistant to respond
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": str(self.device),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def is_ready(self) -> bool:
        """Check if the LLM is ready to use."""
        return self.is_initialized and self.model is not None and self.tokenizer is not None
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        self.logger.info("LLM resources cleaned up")


# Global LLM manager instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def initialize_llm() -> bool:
    """Initialize the global LLM manager."""
    manager = get_llm_manager()
    return manager.initialize()


def cleanup_llm():
    """Clean up the global LLM manager."""
    global _llm_manager
    if _llm_manager is not None:
        _llm_manager.cleanup()
        _llm_manager = None


if __name__ == "__main__":
    # Test the LLM manager
    manager = get_llm_manager()
    
    if manager.initialize():
        print("LLM initialized successfully")
        print(f"Model info: {manager.get_model_info()}")
        
        # Test generation
        test_prompt = "What is Boost.Asio?"
        response = manager.generate_text(test_prompt, max_length=100)
        print(f"Test response: {response}")
        
        manager.cleanup()
    else:
        print("Failed to initialize LLM")
