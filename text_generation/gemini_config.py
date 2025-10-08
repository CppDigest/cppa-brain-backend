"""
Configuration file for the Gemini Chatbot.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Google AI API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model Configuration
DEFAULT_MODEL = "gemini-1.5-flash"  # Can also use "gemini-1.5-pro", "gemini-1.0-pro"
MAX_OUTPUT_TOKENS = 1000
TEMPERATURE = 0.7

# Chat Configuration
MAX_HISTORY_LENGTH = 10  # Number of messages to keep in context
SYSTEM_MESSAGE = "You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, accurate, and engaging responses to user questions and conversations."


def check_available_models():
    """Check which Gemini models are available with your API key."""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Please set your GOOGLE_API_KEY environment variable first!")
        print("You can get an API key from: https://makersuite.google.com/app/apikey")
        return
    
    print("🔍 Checking Available Gemini Models")
    print("=" * 40)
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # List all available models
        models = genai.list_models()
        
        print("📋 Available Models:")
        print("-" * 30)
        
        text_models = []
        vision_models = []
        
        for model in models:
            model_name = model.name.replace('models/', '')
            if 'generateContent' in model.supported_generation_methods:
                if 'vision' in model_name.lower() or 'multimodal' in model_name.lower():
                    vision_models.append(model_name)
                    print(f"🖼️  {model_name} (Vision)")
                else:
                    text_models.append(model_name)
                    print(f"💬 {model_name} (Text)")
        
        print(f"\n📊 SUMMARY")
        print("=" * 20)
        print(f"✅ Text models: {len(text_models)}")
        for model in text_models:
            print(f"   - {model}")
        
        print(f"\n🖼️  Vision models: {len(vision_models)}")
        for model in vision_models:
            print(f"   - {model}")
        
        if text_models:
            print(f"\n🎉 Recommended text model: {text_models[0]}")
            print(f"You can use this model by running:")
            print(f"python gemini_chatbot.py")
            print(f"Then type: model {text_models[0]}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your API key and try again.")

def test_recommended_model():
    """Test the recommended model with a simple conversation."""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return
    
    print(f"\n🧪 Testing Recommended Model")
    print("=" * 35)
    
    try:
        from gemini_chatbot import GeminiChatbot
        
        # Use the most common available model
        chatbot = GeminiChatbot(model="gemini-1.5-flash")
        
        print("Testing with gemini-1.5-flash...")
        response = chatbot.generate_response("Hello! Can you tell me what 2+2 equals?")
        print(f"Bot: {response}")
        
        print("\n🎉 Model test successful!")
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        print("Trying alternative model...")
        
        try:
            chatbot = GeminiChatbot(model="gemini-1.0-pro")
            response = chatbot.generate_response("Hello!")
            print(f"Bot: {response}")
            print("\n🎉 Alternative model test successful!")
        except Exception as e2:
            print(f"❌ Alternative model also failed: {str(e2)}")