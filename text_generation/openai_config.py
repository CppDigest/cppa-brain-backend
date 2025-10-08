"""
Configuration file for the GPT-4 Chatbot.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"  # Can also use "gpt-4", "gpt-4-turbo" (requires special access)
MAX_TOKENS = 500
TEMPERATURE = 0.7

# Chat Configuration
MAX_HISTORY_LENGTH = 10  # Number of messages to keep in context
SYSTEM_MESSAGE = "You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, accurate, and engaging responses to user questions and conversations."


def check_available_models():
    """Check which models are available with your API key."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable first!")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    print("🔍 Checking Available OpenAI Models")
    print("=" * 40)
    
    client = OpenAI(api_key=api_key)
    
    # List of models to test
    models_to_test = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-3.5-turbo-16k"
    ]
    
    available_models = []
    
    for model in models_to_test:
        print(f"Testing {model}...", end=" ")
        
        try:
            # Try to make a simple request to test access
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            print("✅ Available")
            available_models.append(model)
            
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg or "does not exist" in error_msg:
                print("❌ Not available")
            elif "rate_limit" in error_msg.lower():
                print("⏳ Rate limited (but available)")
                available_models.append(model)
            else:
                print(f"⚠️ Error: {error_msg[:50]}...")
    
    print(f"\n📊 RESULTS")
    print("=" * 20)
    
    if available_models:
        print(f"✅ Available models ({len(available_models)}):")
        for model in available_models:
            print(f"   - {model}")
        
        print(f"\n🎉 Recommended model: {available_models[0]}")
        print(f"You can use this model by running:")
        print(f"python openai_chatbot.py")
        print(f"Then type: model {available_models[0]}")
        
    else:
        print("❌ No models available. Please check:")
        print("1. Your API key is correct")
        print("2. You have credits in your OpenAI account")
        print("3. Your account has access to the models")

def test_recommended_model():
    """Test the recommended model with a simple conversation."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return
    
    print(f"\n🧪 Testing Recommended Model")
    print("=" * 35)
    
    try:
        from openai_chatbot import OpenAIChatbot
        
        # Use gpt-3.5-turbo as it's most commonly available
        chatbot = OpenAIChatbot(model="gpt-3.5-turbo")
        
        print("Testing with gpt-3.5-turbo...")
        response = chatbot.generate_response("Hello! Can you tell me what 2+2 equals?")
        print(f"Bot: {response}")
        
        print("\n🎉 Model test successful!")
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")