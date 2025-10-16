"""
Multi-Query Expansion and RAG-Fusion
Implements improvement #3: Multi-Query Expansion (RAG-Fusion)
"""

from typing import List, Dict, Any, Optional, Set
from loguru import logger
import copy

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_config


class QueryExpansion:
    """Generate query variations and expansions for better retrieval coverage."""

    def __init__(self, llm_client=None):
        self.logger = logger.bind(name="QueryExpansion")
        self.llm_client = llm_client
        self.num_paraphrases = get_config(
            "rag.retrieval.multi_query.num_paraphrases", 3
        )

    def generate_paraphrases(
        self, query: str, num_variations: Optional[int] = None
    ) -> List[str]:
        """
        Generate paraphrases of the query using LLM.
        Different phrasings can surface complementary results.
        """
        num_vars = num_variations or self.num_paraphrases

        if self.llm_client is None:
            # Fallback to rule-based variations if no LLM
            return self._rule_based_variations(query)

        try:
            prompt = f"""Generate {num_vars} different ways to ask the same question. 
Each variation should use different words but maintain the same meaning.
Only output the variations, one per line, without numbering or explanation.

Original question: {query}

Variations:"""

            # Call LLM (assuming it has a generate method)
            if hasattr(self.llm_client, "generate"):
                response = self.llm_client.generate(
                    prompt, max_tokens=200, temperature=0.7
                )

                # Parse response
                variations = [
                    line.strip() for line in response.split("\n") if line.strip()
                ]
                variations = [v for v in variations if v and v != query][:num_vars]

                if variations:
                    self.logger.debug(f"Generated {len(variations)} query variations")
                    return [query] + variations

            # Fallback
            return self._rule_based_variations(query)

        except Exception as e:
            self.logger.warning(
                f"⚠️ Failed to generate paraphrases: {e}, using fallback"
            )
            return self._rule_based_variations(query)

    def _rule_based_variations(self, query: str) -> List[str]:
        """Generate simple rule-based query variations."""
        variations = [query]

        # Add question variations
        if not query.strip().endswith("?"):
            variations.append(query + "?")

        # Add "how to" variation if not present
        if "how" not in query.lower():
            variations.append(f"How to {query}")

        # Add "what is" variation
        if query.lower().startswith(("what", "how", "why", "when", "where")):
            # Already a question
            pass
        else:
            variations.append(f"What is {query}")

        return variations[: self.num_paraphrases + 1]

    def expand_with_synonyms(
        self, query: str, synonym_dict: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """
        Expand query with domain-specific synonyms.
        Useful for technical domains with specific terminology.
        """
        if synonym_dict is None:
            # Default C++/Boost synonyms
            synonym_dict = {
                "async": ["asynchronous", "non-blocking", "concurrent"],
                "sync": ["synchronous", "blocking"],
                "function": ["method", "routine", "procedure"],
                "class": ["object", "type", "struct"],
                "error": ["exception", "failure", "issue"],
                "performance": ["speed", "efficiency", "optimization"],
                "thread": ["worker", "concurrent", "parallel"],
            }

        expanded = [query]
        words = query.lower().split()

        for word in words:
            if word in synonym_dict:
                for synonym in synonym_dict[word]:
                    expanded_query = query.replace(word, synonym)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)

        return expanded[:5]  # Limit to 5 expansions

    def pseudo_relevance_feedback(
        self,
        query: str,
        initial_results: List[Dict[str, Any]],
        top_n: int = 3,
        num_terms: int = 5,
    ) -> str:
        """
        Pseudo-Relevance Feedback: Extract important terms from top results to expand query.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            if not initial_results or len(initial_results) < top_n:
                return query

            # Get top documents
            top_docs = [result.text for result in initial_results[:top_n]]

            if not any(top_docs):
                return query

            # Extract important terms using TF-IDF
            vectorizer = TfidfVectorizer(max_features=num_terms, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(top_docs)

            # Get top terms
            feature_names = vectorizer.get_feature_names_out()
            top_terms = feature_names[:num_terms]

            # Add terms to query
            expanded_query = query + " " + " ".join(top_terms)

            self.logger.debug(f"PRF expanded query with terms: {top_terms}")
            return expanded_query

        except Exception as e:
            self.logger.warning(f"⚠️ PRF failed: {e}")
            return query


class RAGFusion:
    """
    RAG-Fusion: Combine results from multiple query variations using Reciprocal Rank Fusion.
    """

    def __init__(self, query_expander: Optional[QueryExpansion] = None):
        self.logger = logger.bind(name="RAGFusion")
        self.query_expander = query_expander or QueryExpansion()
        self.k = 60  # RRF constant (typical value)

    def search_with_fusion(
        self, query: str, search_fn, top_k: int = 10, use_prf: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform RAG-Fusion: generate query variations, search with each, and fuse results.

        Args:
            query: Original query
            search_fn: Function that takes a query and returns List[Dict] with 'id' and 'score'
            top_k: Number of final results to return
            use_prf: Whether to use Pseudo-Relevance Feedback
        """
        try:
            # Generate query variations
            query_variations = self.query_expander.generate_paraphrases(query)
            self.logger.info(
                f"🔄 RAG-Fusion with {len(query_variations)} query variations"
            )

            # Search with each variation
            all_results = {}  # id -> {result_data, rank_positions: []}

            for i, q_var in enumerate(query_variations):
                self.logger.debug(f"  Searching: {q_var[:50]}...")
                results = search_fn(q_var, top_k * 2)  # Get more results for fusion

                # Record rank positions
                for rank, result in enumerate(results, 1):
                    result_id = result.node_id

                    if result_id not in all_results:
                        all_results[result_id] = {
                            "result": result,
                            "rank_positions": [],
                        }

                    all_results[result_id]["rank_positions"].append((i, rank))

            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(all_results)

            # Sort by fused score and return top-k
            fused_results.sort(key=lambda x: x.gen_score, reverse=True)
            final_results = fused_results[:top_k]

            # Apply PRF if enabled
            if use_prf and len(final_results) >= 3:
                expanded_query = self.query_expander.pseudo_relevance_feedback(
                    query, final_results, top_n=3
                )

                if expanded_query != query:
                    self.logger.info(f"🔍 Applying PRF with expanded query")
                    prf_results = search_fn(expanded_query, top_k)

                    # Merge PRF results with fusion results
                    final_results = self._merge_results(
                        final_results, prf_results, top_k
                    )

            self.logger.info(f"✅ RAG-Fusion complete: {len(final_results)} results")
            return final_results

        except Exception as e:
            self.logger.error(f"❌ RAG-Fusion failed: {e}")
            # Fallback to simple search
            return search_fn(query, top_k)

    def _get_result_id(self, result: Dict[str, Any]) -> str:
        """Get unique identifier for a result."""
        # Try different ID fields
        for id_field in ["id", "doc_id", "chunk_id", "source_file"]:
            if id_field in result:
                return str(result[id_field])

        # Fallback: hash of text
        text = result.text
        return str(hash(text))

    def _reciprocal_rank_fusion(
        self, all_results: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine rankings.
        RRF score = sum(1 / (k + rank)) across all queries where doc appears.
        """
        fused = []

        for result_id, data in all_results.items():
            # Calculate RRF score
            rrf_score = 0.0
            for query_idx, rank in data["rank_positions"]:
                rrf_score += 1.0 / (self.k + rank)

            result = copy.deepcopy(data["result"])
            result.gen_score = rrf_score
            # result["appeared_in_queries"] = len(data["rank_positions"])
            fused.append(result)

        return fused

    def _merge_results(
        self, fusion_results: List[Dict], prf_results: List[Dict], top_k: int
    ) -> List[Dict[str, Any]]:
        """Merge fusion results with PRF results, removing duplicates."""
        seen_ids = set()
        merged = []

        # Add fusion results first
        for result in fusion_results:
            result_id = self._get_result_id(result)
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                merged.append(result)

        # Add PRF results if not already present
        for result in prf_results:
            result_id = self._get_result_id(result)
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                merged.append(result)

                if len(merged) >= top_k:
                    break

        return merged[:top_k]


class QueryAnalyzer:
    """Analyze queries to determine expansion strategy."""

    def __init__(self):
        self.logger = logger.bind(name="QueryAnalyzer")

    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics."""
        analysis = {
            "length": len(query.split()),
            "is_question": query.strip().endswith("?"),
            "has_code": any(
                keyword in query.lower()
                for keyword in ["function", "class", "method", "code"]
            ),
            "is_short": len(query.split()) <= 3,
            "complexity": self._estimate_complexity(query),
        }

        # Determine recommended strategy
        if analysis["is_short"]:
            analysis["recommended_strategy"] = "aggressive_expansion"
        elif analysis["has_code"]:
            analysis["recommended_strategy"] = "technical_expansion"
        elif analysis["is_question"]:
            analysis["recommended_strategy"] = "paraphrase_expansion"
        else:
            analysis["recommended_strategy"] = "standard_expansion"

        return analysis

    def _estimate_complexity(self, query: str) -> str:
        """Estimate query complexity."""
        word_count = len(query.split())

        if word_count <= 3:
            return "simple"
        elif word_count <= 10:
            return "medium"
        else:
            return "complex"
