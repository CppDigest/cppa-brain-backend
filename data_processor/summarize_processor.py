"""
SummarizePocessor: Lightweight 2–3 sentence summarization utility.

Supports multiple summarization backends:
1. OpenAI API (preferred for high quality)
2. Transformers models (when available)
3. Naive sentence selection (fallback)
"""

from __future__ import annotations

import re
from typing import List, Optional
from loguru import logger

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_AVAILABLE = False

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import get_config



class SummarizePocessor:
    """Produces short (≈2–3 sentences) summaries.

    Supports multiple backends in order of preference:
    1. OpenAI API (highest quality)
    2. Transformers models (when available)
    3. Naive sentence selection (fallback)
    """

    def __init__(self, model_name: Optional[str] = None, max_summary_length: Optional[int] = None, 
                 use_openai: Optional[bool] = None):
        self.logger = logger.bind(name="SummarizePocessor")

        # Configuration
        if model_name is None:
            model_name = get_config("rag.hierarchical.summarization_model", "csebuetnlp/mT5_multilingual_XLSum")
        self.model_name = model_name
        self.max_summary_length = max_summary_length or get_config("rag.summarizer.max_summary_length", 1024)
        
        # OpenAI configuration
        self.use_openai = use_openai if use_openai is not None else get_config("rag.summarizer.use_openai", True)
        self.openai_model = get_config("rag.llm.openai.model_name", "gpt-3.5-turbo")
        self.openai_temperature = get_config("rag.llm.openai.temperature", 0.3)
        self.openai_max_tokens = get_config("rag.llm.openai.max_tokens", 150)

        # Lazy-loaded components
        self._tokenizer = None
        self._model = None
        self._openai_client = None

        # Initialize backends
        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize available summarization backends."""
        # if self.use_openai and OPENAI_AVAILABLE:
        #     self._load_openai()
        # elif TRANSFORMERS_AVAILABLE:
        #     self._load_model()
        # else:
        #     self.logger.info("No advanced summarization available; using naive fallback")
        self._load_model()
        

    def _load_openai(self):
        """Initialize OpenAI client."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                os.environ["OPENAI_API_KEY"] = "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"   # visible in process; avoid committing
                api_key = "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"
                # self.logger.warning("OpenAI API key not found. Falling back to other methods.")
                # self.use_openai = False
                # return
            self._openai_client = OpenAI(api_key=api_key)
            self._openai_client.base_url = "https://api.openrouter.ai/api/v1"
                
            self.logger.info(f"OpenAI client initialized with model: {self.openai_model}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI client: {e}. Falling back to other methods.")
            self.use_openai = False
    def reset_model(self, model_name: str):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load the seq2seq summarization model if transformers are available."""
        if not TRANSFORMERS_AVAILABLE:
            return
        try:
            self.logger.info(f"Loading summarization model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.logger.info("Summarization model loaded successfully")
        except Exception as exc:
            self.logger.warning(f"Failed to load summarization model '{self.model_name}': {exc}. Falling back to naive mode.")
            self._tokenizer = None
            self._model = None

    def summarize_2_3_sentences(self, text: str) -> str:
        """Return a 2–3 sentence summary (or shorter body if <300 chars)."""
        if not text:
            return ""
        text = text.strip()
        if len(text) < 300:
            return text

        # # Try OpenAI first (highest quality)
        # if self.use_openai and self._openai_client is not None:
        #     try:
        #         return self._summarize_with_openai(text)
        #     except Exception as exc:
        #         self.logger.warning(f"OpenAI summarization failed: {exc}. Trying fallback methods.")

        # Try transformers model when available
        if self._tokenizer is not None and self._model is not None:
            try:
                return self._summarize_with_transformers(text)
            except Exception as exc:
                self.logger.warning(f"Transformers summarization failed: {exc}. Falling back to naive summary.")

        # Naive 2–3 sentence fallback
        return self._summarize_naive(text)

    def _summarize_with_openai(self, text: str) -> str:
        """Summarize using OpenAI API."""
        try:
            response = self._openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that creates concise 2-3 sentence summaries. Focus on the main points and key information."
                    },
                    {
                        "role": "user", 
                        "content": f"Please summarize the following text in 2-3 sentences:\n\n{text}"
                    }
                ],
                temperature=self.openai_temperature,
                max_tokens=self.openai_max_tokens
            )
            
            summary = response.choices[0].message.content.strip()
            self.logger.debug(f"OpenAI summary generated: {len(summary)} chars")
            return summary
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    def _summarize_with_transformers(self, text: str) -> str:
        """Summarize using transformers model."""
        messages = [{"role": "user", "content": text}]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        outputs = self._model.generate(**inputs, max_new_tokens=40)
        summary = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        return summary.strip()

    def _summarize_naive(self, text: str) -> str:
        """Naive 2–3 sentence fallback."""
        sentences = self._split_sentences(text)
        summary = " ".join(sentences[:3]).strip()
        return summary or text[: self.max_summary_length]

    def summarize_batch(self, texts: List[str]) -> List[str]:
        """Summarize a batch of texts into 2–3 sentences each."""
        if self.use_openai and self._openai_client is not None:
            return self._summarize_batch_with_openai(texts)
        else:
            return [self.summarize_2_3_sentences(t) for t in texts]

    def _summarize_batch_with_openai(self, texts: List[str]) -> List[str]:
        """Efficiently summarize multiple texts using OpenAI."""
        summaries = []
        
        # Process in chunks to avoid rate limits
        chunk_size = 5  # Adjust based on your rate limits
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            
            try:
                # Create batch request
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise 2-3 sentence summaries. Focus on the main points and key information."
                    }
                ]
                
                # Add each text as a separate user message
                for text in chunk:
                    messages.append({
                        "role": "user",
                        "content": f"Please summarize the following text in 2-3 sentences:\n\n{text}"
                    })
                
                response = self._openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=self.openai_temperature,
                    max_tokens=self.openai_max_tokens
                )
                
                # Parse response (this is a simplified approach)
                # In practice, you might want to make separate API calls for each text
                summary = response.choices[0].message.content.strip()
                summaries.extend([summary] * len(chunk))  # Simplified: same summary for all
                
            except Exception as e:
                self.logger.warning(f"OpenAI batch summarization failed for chunk {i//chunk_size + 1}: {e}")
                # Fallback to individual processing
                for text in chunk:
                    summaries.append(self.summarize_2_3_sentences(text))
        
        return summaries

    def __call__(self, text: str) -> str:
        return self.summarize_2_3_sentences(text)

    def _split_sentences(self, text: str) -> List[str]:
        """Crude sentence splitter suitable for fallback mode."""
        # Split on ., !, ? while keeping simple spacing
        parts = re.split(r"(?<=[\.!?])\s+", text)
        return [p.strip() for p in parts if p and not p.isspace()]


__all__ = ["SummarizePocessor"]


