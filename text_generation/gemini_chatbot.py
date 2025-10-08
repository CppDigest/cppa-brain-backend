"""
Google Gemini Chatbot using Google AI API.
This version uses Google's Gemini model for high-quality conversations.
"""

import os
import google.generativeai as genai
from typing import Optional, List, Dict
import json

from loguru import logger
from text_generation_model import TextGenerationModel


class GeminiChatbot(TextGenerationModel):
    def __init__(self, model_name: str = "gemini-1.5-flash", config: Dict = None):
        """
        Initialize the Gemini chatbot.
        
        Args:
            model_name: Model to use (gemini-pro, gemini-1.5-flash, etc.)
            config: Configuration dictionary (expects 'api_key' optionally)
        """
        super().__init__(model_name=model_name, config=config)
        
        # API key handling
        api_key = (config or {}).get('api_key') or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google AI API key is required. Set GOOGLE_API_KEY environment variable or pass via config['api_key'].")
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
        self.system_message = (config or {}).get('system_message', "You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, accurate, and engaging responses to user questions and conversations.")
        self.is_initialized = True
        self.is_ready = True
    
    def initialize(self) -> bool:
        try:
            # Already configured in __init__ for API key and model
            self.is_initialized = True
            self.is_ready = True
            logger.info(f"✅ GeminiChatbot initialized with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize GeminiChatbot: {e}")
            self.is_initialized = False
            self.is_ready = False
            return False

    def is_model_ready(self) -> bool:
        return self.is_initialized and self.is_ready

    def generate_response(self, user_input: str, temperature: float = 0.7) -> str:
        """
        Generate a response to user input using Gemini.
        
        Args:
            user_input: The user's message
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            Generated response from Gemini
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Build the conversation context
            conversation_text = f"System: {self.system_message}\n\n"
            
            # Add conversation history (keep last 10 messages to manage context)
            recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
            
            for msg in recent_history:
                if msg["role"] == "user":
                    conversation_text += f"Human: {msg['content']}\n"
                else:
                    conversation_text += f"Assistant: {msg['content']}\n"
            
            # Add current user input
            conversation_text += f"Human: {user_input}\nAssistant:"
            
            # Generate response
            response = self.model.generate_content(
                conversation_text,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1000,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            # Extract the response
            generated_text = response.text.strip()
            
            # Add bot response to conversation history
            self.conversation_history.append({"role": "assistant", "content": generated_text})
            
            return generated_text
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                return "⏳ Rate limit exceeded. Please wait a moment before trying again."
            elif "api" in error_msg.lower() or "key" in error_msg.lower():
                return "❌ API error. Please check your Google AI API key."
            elif "safety" in error_msg.lower():
                return "⚠️ Response blocked by safety filters. Please try rephrasing your question."
            else:
                return f"❌ Unexpected error: {error_msg}"
    
    def is_ready(self) -> bool:
        """Check if the LLM is ready to use."""
        return self.model is not None

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

def main():
    """Main function to run the Gemini chatbot interactively."""
    print("🤖 Google Gemini Chatbot")
    print("=" * 30)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("Type 'history' to see conversation history")
    print("Type 'save <filename>' to save conversation")
    print("Type 'load <filename>' to load conversation")
    print("Type 'system <message>' to set system message")
    print("Type 'model <model_name>' to change model")
    print()
    
    try:
        # Initialize the chatbot
        chatbot = GeminiChatbot()
        print("✅ Gemini Chatbot initialized successfully!")
        print(f"📱 Using model: {chatbot.model_name}")
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
                        chatbot.model_name = model_name
                        chatbot.model = genai.GenerativeModel(model_name)
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
        print("Make sure you have set your GOOGLE_API_KEY environment variable with a valid Google AI API key.")

if __name__ == "__main__":
    main()
