import os
import requests
import json
from typing import Optional, Dict
from loguru import logger
from text_generation_model import TextGenerationModel
from ollama_config import API_TIMEOUT  # reuse timeout constant

# Defaults for HF if not provided via config
DEFAULT_HF_MODEL = "microsoft/DialoGPT-medium"
MAX_LENGTH = 500
TEMPERATURE = 0.7


class HuggingFaceChatbot(TextGenerationModel):
    def __init__(self, model_name: str = DEFAULT_HF_MODEL, config: Dict = None):
        """
        Initialize the Hugging Face chatbot.
        
        Args:
            model_name: Model to use for chat generation
            config: Configuration dict, expects 'api_key'
        """
        super().__init__(model_name=model_name, config=config)
        self.api_key = (config or {}).get('api_key') or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("Hugging Face API key is required. Set HF_TOKEN environment variable or pass api_key parameter.")
        
        self.model = model_name
        # Use the correct API endpoint format
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.conversation_history = []
        self.is_initialized = True
        self.is_ready = True

    def initialize(self) -> bool:
        try:
            self.is_initialized = True
            self.is_ready = True
            logger.info(f"✅ HuggingFaceChatbot initialized with model: {self.model}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize HuggingFaceChatbot: {e}")
            self.is_initialized = False
            self.is_ready = False
            return False

    def is_model_ready(self) -> bool:
        return self.is_initialized and self.is_ready
    
    def generate_response(self, user_input: str, max_length: int = MAX_LENGTH) -> str:
        """
        Generate a response to user input using Hugging Face API.
        
        Args:
            user_input: The user's message
            max_length: Maximum length of generated response
            
        Returns:
            Generated response from the model
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Prepare the payload for the API
        payload = {
            "inputs": user_input,
            "parameters": {
                "max_length": max_length,
                "temperature": TEMPERATURE,
                "do_sample": True,
                "return_full_text": False,
                "pad_token_id": 50256  # GPT-2 pad token
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            
            # Handle different response status codes
            if response.status_code == 503:
                return "The model is currently loading. Please wait a moment and try again."
            elif response.status_code == 404:
                return f"Model '{self.model}' not found. Please try a different model."
            elif response.status_code == 401:
                return "Authentication failed. Please check your API key."
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text based on response format
            generated_text = ""
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    generated_text = result[0]["generated_text"].strip()
                elif "text" in result[0]:
                    generated_text = result[0]["text"].strip()
                else:
                    # Handle different response formats
                    generated_text = str(result[0]).strip()
            elif isinstance(result, dict):
                if "generated_text" in result:
                    generated_text = result["generated_text"].strip()
                elif "text" in result:
                    generated_text = result["text"].strip()
            
            # Clean up the response
            if generated_text:
                # Remove the original input if it's included in the response
                if user_input in generated_text:
                    generated_text = generated_text.replace(user_input, "").strip()
                # Remove any special tokens
                generated_text = generated_text.replace("<|endoftext|>", "").strip()
            else:
                generated_text = "I'm sorry, I couldn't generate a response right now."
            
            # Add bot response to conversation history
            self.conversation_history.append({"role": "assistant", "content": generated_text})
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Hugging Face API: {str(e)}"
            print(error_msg)
            return "I'm sorry, I'm having trouble connecting to the AI service right now."
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)
            return "I'm sorry, something went wrong. Please try again."
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_history(self):
        """Get the current conversation history."""
        return self.conversation_history.copy()

def main():
    """Main function to run the chatbot interactively."""
    print("🤖 Hugging Face Chatbot")
    print("=" * 30)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("Type 'history' to see conversation history")
    print()
    
    try:
        # Initialize the chatbot
        chatbot = HuggingFaceChatbot()
        print("✅ Chatbot initialized successfully!")
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
                    for msg in history:
                        role = "You" if msg["role"] == "user" else "Bot"
                        print(f"{role}: {msg['content']}")
                else:
                    print("📜 No conversation history yet.")
                print()
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
        print("Make sure you have set your HF_TOKEN environment variable with a valid Hugging Face API key.")

if __name__ == "__main__":
    main()
