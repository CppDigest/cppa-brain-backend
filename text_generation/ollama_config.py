"""
Configuration file for the Ollama Chatbot.
"""

import os
import requests
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model Configuration
DEFAULT_MODEL = "gemma3:12b"  # Can also use "llama2", "codellama", "mistral", "phi", etc.
TEMPERATURE = 0.7

# Chat Configuration
MAX_HISTORY_LENGTH = 10  # Number of messages to keep in context
SYSTEM_MESSAGE = "You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, accurate, and engaging responses to user questions and conversations."


def get_available_models():
    """
    Get list of available models from Ollama server.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            logger.info(f"Found {len(models)} available models: {models}")
            return models
        else:
            logger.warning(f"Failed to get models from Ollama: {response.status_code}")
            return [DEFAULT_MODEL]  # Fallback to default model
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return [DEFAULT_MODEL]  # Fallback to default model


def get_available_models_with_size():
    """
    Get list of available models from Ollama server with size information, sorted by size.
    Returns list of dicts with 'name' and 'size' keys, sorted by size (smallest first).
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            
            # Extract model info and sort by size
            model_info = []
            for model in models:
                model_info.append({
                    "name": model["name"],
                    "size": model.get("size", 0),
                    "parameter_size": model.get("details", {}).get("parameter_size", "Unknown")
                })
            
            # Sort by size (smallest first)
            model_info.sort(key=lambda x: x["size"])
            
            logger.info(f"Found {len(model_info)} available models, sorted by size")
            return model_info
        else:
            logger.warning(f"Failed to get models from Ollama: {response.status_code}")
            return [{"name": DEFAULT_MODEL, "size": 0, "parameter_size": "Unknown"}]  # Fallback
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return [{"name": DEFAULT_MODEL, "size": 0, "parameter_size": "Unknown"}]  # Fallback


def get_llm_models_sorted_by_size():
    """
    Get list of LLM models (excluding embedding models) sorted by size.
    Returns list of model names sorted by size (smallest first).
    """
    try:
        models_with_size = get_available_models_with_size()
        
        # Filter for LLM models (exclude embedding models)
        llm_models = []
        for model in models_with_size:
            model_name = model["name"]
            # Exclude embedding models
            if not any(embed_word in model_name.lower() for embed_word in ["embed", "embedding"]):
                llm_models.append(model)
        
        # If no LLM models found, return some common fallback options
        if not llm_models:
            llm_models = [
                {"name": "gemma3:12b", "size": 0, "parameter_size": "Unknown"},
                {"name": "llama2", "size": 0, "parameter_size": "Unknown"},
                {"name": "mistral", "size": 0, "parameter_size": "Unknown"}
            ]
        
        logger.info(f"Found {len(llm_models)} LLM models, sorted by size")
        return llm_models
    except Exception as e:
        logger.error(f"Error getting LLM models: {e}")
        # Return fallback LLM models on error
        return [
            {"name": "gemma3:12b", "size": 0, "parameter_size": "Unknown"},
            {"name": "llama2", "size": 0, "parameter_size": "Unknown"},
            {"name": "mistral", "size": 0, "parameter_size": "Unknown"}
        ]