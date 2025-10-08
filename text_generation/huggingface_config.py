"""
Configuration file for the Hugging Face Chatbot.
Copy this file to .env and add your actual API key.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face API Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model Configuration
DEFAULT_MODEL = "EleutherAI/gpt-j-6B"  # More reliable model
MAX_LENGTH = 100
TEMPERATURE = 0.7

# API Configuration
API_TIMEOUT = 30
