"""
GPT-4 Chatbot using OpenAI API.
This version uses OpenAI's GPT-4 model for high-quality conversations.
"""

import os
from openai import OpenAI
from typing import Optional, List, Dict
import json

from loguru import logger
from text_generation_model import TextGenerationModel


class OpenAIChatbot(TextGenerationModel):
    def __init__(self, model_name: str = "gpt-4-turbo", config: Dict = None):
        """
        Initialize the Openai Chatbot.
        
        Args:
            model_name: Model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.)
            config: Configuration dict, expects 'api_key'
        """
        super().__init__(model_name=model_name, config=config)
        self.api_key = (config or {}).get('api_key') or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client with the new API
        self.client = OpenAI(api_key=self.api_key)
        
        self.model = model_name
        self.conversation_history = []
        self.system_message = (config or {}).get('system_message', "You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, accurate, and engaging responses to user questions and conversations.")
        self.is_initialized = True
        self.is_ready = True
    
    def initialize(self) -> bool:
        try:
            # Already initialized client
            self.is_initialized = True
            self.is_ready = True
            logger.info(f"✅ OpenAIChatbot initialized with model: {self.model}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAIChatbot: {e}")
            self.is_initialized = False
            self.is_ready = False
            return False

    def is_model_ready(self) -> bool:
        return self.is_initialized and self.is_ready

    def generate_response(self, user_input: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate a response to user input using GPT-4.
        
        Args:
            user_input: The user's message
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response from GPT-4
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Prepare messages for API call
            messages = [{"role": "system", "content": self.system_message}]
            
            # Add conversation history (keep last 10 messages to manage context)
            recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
            messages.extend(recent_history)
            
            # Call OpenAI API using the new client
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Extract the response
            generated_text = response.choices[0].message.content.strip()
            
            # Add bot response to conversation history
            self.conversation_history.append({"role": "assistant", "content": generated_text})
            
            return generated_text
            
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
                return "❌ Authentication failed. Please check your OpenAI API key."
            elif "rate limit" in error_msg.lower():
                return "⏳ Rate limit exceeded. Please wait a moment before trying again."
            elif "api" in error_msg.lower():
                return f"❌ OpenAI API error: {error_msg}"
            else:
                return f"❌ Unexpected error: {error_msg}"
    
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
    """Main function to run the Openai Chatbot interactively."""
    print("🤖 Openai Chatbot")
    print("=" * 20)
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
        chatbot = OpenAIChatbot()
        print("✅ Openai Chatbot initialized successfully!")
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
        print("Make sure you have set your OPENAI_API_KEY environment variable with a valid OpenAI API key.")

if __name__ == "__main__":
    main()
