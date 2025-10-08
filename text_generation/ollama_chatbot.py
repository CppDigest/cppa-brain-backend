"""
Ollama Chatbot using local Ollama models.
This version uses Ollama to run LLMs locally on your machine.
"""

import os
import requests
import json
from typing import Optional, List, Dict
import time
from loguru import logger

from text_generation.text_generation_model import TextGenerationModel

class OllamaChatbot(TextGenerationModel):
    def __init__(self, model_name: str = "gemma3:12b", config: Dict = None):
        """
        Initialize the Ollama chatbot.
        
        Args:
            model_name: Ollama model to use (llama2, codellama, mistral, etc.)
            config: Configuration dictionary
        """
        # Initialize parent class
        super().__init__(model_name=model_name, config=config)
        
        # Ollama-specific configuration
        self.base_url = config.get('base_url', 'http://localhost:11434') if config else 'http://localhost:11434'
        self.conversation_history = []
        self.system_message = config.get('system_message', "You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, accurate, and engaging responses to user questions and conversations.") if config else "You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, accurate, and engaging responses to user questions and conversations."
        self.history_length = config.get('history_length', 3) if config else 3
        
        # Initialize the model
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the Ollama chatbot.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._test_connection()
            self.is_initialized = True
            self.is_ready = True
            self.logger.info(f"✅ OllamaChatbot initialized with model: {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OllamaChatbot: {e}")
            self.is_initialized = False
            self.is_ready = False
            return False
    
    def generate_response(self, user_input: str, **kwargs) -> str:
        """
        Generate a response from the Ollama model.
        
        Args:
            user_input: Input text from user
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated response text
        """
        try:
            # Use the existing generate method
            return self.generate(user_input, **kwargs)
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def is_model_ready(self) -> bool:
        """
        Check if the model is ready for use.
        
        Returns:
            True if model is ready, False otherwise
        """
        return self.is_ready and self.is_initialized
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to Ollama server at {self.base_url}")
                self._check_model_availability()
            else:
                print(f"⚠️ Ollama server responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama server at {self.base_url}")
            print("Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {str(e)}")
    
    def _check_model_availability(self):
        """Check if the specified model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model_name in model_names:
                    print(f"✅ Model '{self.model_name}' is available")
                else:
                    print(f"⚠️ Model '{self.model_name}' not found. Available models:")
                    for name in model_names:
                        print(f"   - {name}")
                    print(f"\nTo install a model, run: ollama pull {self.model_name}")
        except Exception as e:
            print(f"❌ Error checking models: {str(e)}")
    
    def generate_response(self, user_input: str, temperature: float = 0.7) -> str:
        """
        Generate a response to user input using Ollama.
        
        Args:
            user_input: The user's message
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            Generated response from Ollama
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Build the conversation context
            messages = [{"role": "system", "content": self.system_message},
                        {"role": "user", "content": user_input}]
            
            # Add conversation history (keep last 10 messages to manage context)
            recent_history = self.conversation_history[-self.history_length:] if len(self.conversation_history) > self.history_length else self.conversation_history
            # messages.extend(recent_history)
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=240  # Ollama can take time to respond
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["message"]["content"].strip()
                
                # Add bot response to conversation history
                self.conversation_history.append({"role": "assistant", "content": generated_text})
                
                return generated_text
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', 'Unknown error')}"
                    except:
                        error_msg += f" - {response.text}"
                return f"❌ {error_msg}"
                
        except requests.exceptions.Timeout:
            return "⏳ Request timed out. The model might be loading or taking too long to respond."
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama server. Make sure Ollama is running: ollama serve"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    
    def is_ready(self) -> bool:
        """Check if the LLM is ready to use."""
        return self.model_name is not None
    
    def set_system_message(self, message: str):
        """Set the system message that defines the AI's behavior."""
        self.system_message = message
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_history(self):
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def save_conversation(self, filename: str):
        """Save conversation history to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            print(f"💾 Conversation saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving conversation: {str(e)}")
    
    def load_conversation(self, filename: str):
        """Load conversation history from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"📂 Conversation loaded from {filename}")
        except Exception as e:
            print(f"❌ Error loading conversation: {str(e)}")
    
    def list_available_models(self):
        """List all available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                print("📋 Available Ollama models:")
                for model in models:
                    name = model["name"]
                    size = model.get("size", 0)
                    size_gb = size / (1024**3) if size > 0 else 0
                    print(f"   - {name} ({size_gb:.1f} GB)")
                return [model["name"] for model in models]
            else:
                print(f"❌ Error listing models: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error listing models: {str(e)}")
            return []

def main():
    """Main function to run the Ollama chatbot interactively."""
    print("🤖 Ollama Local Chatbot")
    print("=" * 30)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("Type 'history' to see conversation history")
    print("Type 'save <filename>' to save conversation")
    print("Type 'load <filename>' to load conversation")
    print("Type 'system <message>' to set system message")
    print("Type 'model <model_name>' to change model")
    print("Type 'models' to list available models")
    print()
    
    try:
        # Initialize the chatbot
        chatbot = OllamaChatbot()
        print(f"📱 Using model: {chatbot.model}")
        print()
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🗑️ Conversation history cleared!")
                continue
            elif user_input.lower() == 'history':
                history = chatbot.get_history()
                if history:
                    print("\n📜 Conversation History:")
                    for i, msg in enumerate(history, 1):
                        role = "You" if msg["role"] == "user" else "Bot"
                        print(f"{i}. {role}: {msg['content']}")
                else:
                    print("📜 No conversation history yet.")
                print()
                continue
            elif user_input.lower() == 'models':
                chatbot.list_available_models()
                continue
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip()
                if filename:
                    chatbot.save_conversation(filename)
                else:
                    print("❌ Please provide a filename: save <filename>")
                continue
            elif user_input.lower().startswith('load '):
                filename = user_input[5:].strip()
                if filename:
                    chatbot.load_conversation(filename)
                else:
                    print("❌ Please provide a filename: load <filename>")
                continue
            elif user_input.lower().startswith('system '):
                message = user_input[7:].strip()
                if message:
                    chatbot.set_system_message(message)
                    print(f"✅ System message updated: {message}")
                else:
                    print("❌ Please provide a system message: system <message>")
                continue
            elif user_input.lower().startswith('model '):
                model_name = user_input[6:].strip()
                if model_name:
                    try:
                        chatbot.model = model_name
                        print(f"✅ Model changed to: {model_name}")
                    except Exception as e:
                        print(f"❌ Error changing model: {str(e)}")
                else:
                    print("❌ Please provide a model name: model <model_name>")
                continue
            elif not user_input:
                continue
            
            # Generate and display response
            print("Bot: ", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print(response)
            print()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure Ollama is running: ollama serve")

if __name__ == "__main__":
    main()
