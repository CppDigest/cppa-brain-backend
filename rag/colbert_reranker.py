"""
ColBERT Late Interaction Reranker
Implements improvement #12: Late Interaction Option (ColBERT)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger
from utils.config import get_config


class ColBERTReranker:
    """
    ColBERT (Contextualized Late Interaction over BERT) reranker.
    Uses late interaction between query and document token embeddings
    for better ranking accuracy, especially on code-heavy text.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.logger = logger.bind(name="ColBERTReranker")
        self.model_name = model_name or get_config(
            "rag.reranker.colbert_model", "colbert-ir/colbertv2.0"
        )
        self.model = None
        self.enabled = get_config("rag.reranker.colbert_enabled", False)
        self.max_latency_ms = get_config("rag.reranker.colbert_max_latency_ms", 300)

    def load_model(self):
        """Load ColBERT model."""
        if self.model is not None:
            return

        try:
            self.logger.info(f"📥 Loading ColBERT model: {self.model_name}")

            # Try to import colbert
            try:
                from colbert.modeling.colbert import ColBERT
                from colbert.infra import ColBERTConfig

                config = ColBERTConfig()
                self.model = ColBERT.from_pretrained(self.model_name, config)

                self.logger.info("✅ ColBERT model loaded")

            except ImportError:
                self.logger.warning(
                    "⚠️ ColBERT not installed. Install with: pip install colbert-ai"
                )
                self.enabled = False
                return

                # Fallback to sentence-transformers based implementation
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)
                self.logger.info("✅ Using sentence-transformers fallback")

        except Exception as e:
            self.logger.error(f"❌ Failed to load ColBERT model: {e}")
            self.enabled = False

    def rerank(
        self, query: str, results: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using ColBERT late interaction.

        Args:
            query: Search query
            results: List of candidate results (typically top 100-200)
            top_k: Number of results to return after reranking

        Returns:
            Reranked results
        """
        if not self.enabled or not results:
            return results[:top_k]

        if self.model is None:
            self.load_model()

        if self.model is None:
            return results[:top_k]

        try:
            import time

            start_time = time.time()

            # Extract texts
            texts = [r.get("text", "") for r in results]

            # Compute late interaction scores
            scores = self._compute_scores(query, texts)

            # Check latency
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.max_latency_ms:
                self.logger.warning(
                    f"⚠️ ColBERT exceeded latency budget: {elapsed_ms:.1f}ms > {self.max_latency_ms}ms"
                )

            # Update scores and sort
            reranked_results = []
            for result, score in zip(results, scores):
                result_copy = result.copy()
                result_copy["colbert_score"] = float(score)
                result_copy["original_score"] = result.get("score", 0.0)
                result_copy["score"] = float(score)
                reranked_results.append(result_copy)

            # Sort by new scores
            reranked_results.sort(key=lambda x: x["score"], reverse=True)

            self.logger.debug(f"✅ ColBERT reranking complete in {elapsed_ms:.1f}ms")

            return reranked_results[:top_k]

        except Exception as e:
            self.logger.error(f"❌ ColBERT reranking failed: {e}")
            return results[:top_k]

    def _compute_scores(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Compute late interaction scores.

        Late interaction: Each query token attends to all document tokens.
        Score = sum over query tokens of max similarity to any document token.
        """
        try:
            # Get token-level embeddings
            query_embeddings = self._encode_tokens(query)  # (num_query_tokens, dim)
            doc_embeddings = [
                self._encode_tokens(doc) for doc in documents
            ]  # List of (num_doc_tokens, dim)

            scores = []

            for doc_emb in doc_embeddings:
                # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
                similarity_matrix = np.matmul(query_embeddings, doc_emb.T)

                # Late interaction: for each query token, take max similarity to any doc token
                max_similarities = np.max(similarity_matrix, axis=1)

                # Sum over query tokens
                score = np.sum(max_similarities)
                scores.append(score)

            return np.array(scores)

        except Exception as e:
            self.logger.error(f"❌ Score computation failed: {e}")
            # Return fallback scores
            return np.array([0.0] * len(documents))

    def _encode_tokens(self, text: str) -> np.ndarray:
        """
        Encode text into token-level embeddings.

        Returns:
            Array of shape (num_tokens, embedding_dim)
        """
        try:
            # Check if we have true ColBERT model
            if hasattr(self.model, "tokenizer"):
                # ColBERT model
                tokens = self.model.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )

                with np.no_grad():
                    import torch

                    outputs = self.model(**tokens)
                    embeddings = outputs.last_hidden_state[0]  # (seq_len, dim)

                return embeddings.cpu().numpy()

            else:
                # Sentence-transformers fallback
                # Approximate token-level embeddings by splitting on whitespace
                words = text.split()
                if not words:
                    return np.zeros((1, 384))  # Default dimension

                # Encode each word separately (approximation)
                word_embeddings = self.model.encode(words)
                return word_embeddings

        except Exception as e:
            self.logger.error(f"❌ Token encoding failed: {e}")
            return np.zeros((1, 384))  # Fallback


class HybridColBERTReranker:
    """
    Hybrid reranker that combines cross-encoder and ColBERT.
    Uses cross-encoder for fast first-pass, ColBERT for final ranking.
    """

    def __init__(
        self, cross_encoder, colbert_reranker: Optional[ColBERTReranker] = None
    ):
        self.logger = logger.bind(name="HybridColBERTReranker")
        self.cross_encoder = cross_encoder
        self.colbert = colbert_reranker or ColBERTReranker()

        # Configuration
        self.use_colbert_for_code = get_config(
            "rag.reranker.colbert_for_code_only", True
        )
        self.cross_encoder_k = get_config("rag.reranker.cross_encoder_k", 20)
        self.colbert_k = get_config("rag.reranker.colbert_k", 5)

    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Two-stage reranking:
        1. Cross-encoder on all results -> top K1
        2. ColBERT on top K1 -> top K2
        """
        if not results:
            return []

        # Stage 1: Cross-encoder reranking
        self.logger.debug("🔄 Stage 1: Cross-encoder reranking...")
        stage1_results = self.cross_encoder.rerank(query, results)
        stage1_results = stage1_results[: self.cross_encoder_k]

        # Decide whether to use ColBERT
        use_colbert = True

        if self.use_colbert_for_code:
            # Only use ColBERT if query or results contain code
            has_code = self._has_code(query) or any(
                self._has_code(r.text) for r in stage1_results[:5]
            )

            if not has_code:
                use_colbert = False
                self.logger.debug("⏩ Skipping ColBERT (no code detected)")

        # Stage 2: ColBERT reranking (if enabled)
        if use_colbert and self.colbert.enabled:
            self.logger.debug("🔄 Stage 2: ColBERT reranking...")
            final_results = self.colbert.rerank(query, stage1_results, self.colbert_k)
        else:
            final_results = stage1_results[: self.colbert_k]

        return final_results

    def _has_code(self, text: str) -> bool:
        """Check if text contains code."""
        code_indicators = [
            "()",
            "::",
            "->",
            "template<",
            "class ",
            "struct ",
            "def ",
            "import ",
            "#include",
            "```",
        ]
        return any(indicator in text for indicator in code_indicators)
