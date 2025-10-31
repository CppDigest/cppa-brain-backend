"""
Query Intent Classification and Routing
Implements improvement #13: Query Intent Routing
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
from loguru import logger
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_config


class QueryIntent(Enum):
    """Types of query intents."""

    DOCUMENTATION = "documentation"  # Looking for official docs/API reference
    HOW_TO = "how_to"  # How to accomplish a task
    DEBUG = "debug"  # Debugging/error resolution
    CODE_EXAMPLE = "code_example"  # Looking for code examples
    MAIL_HISTORY = "mail_history"  # Searching email/discussion history
    CONCEPT = "concept"  # Understanding a concept
    COMPARISON = "comparison"  # Comparing options
    UNKNOWN = "unknown"


class QueryIntentClassifier:
    """Classify query intent to route to appropriate retrieval strategies."""

    def __init__(self, llm_client=None):
        self.logger = logger.bind(name="QueryIntentClassifier")
        self.llm_client = llm_client

        # Intent patterns (rule-based fallback)
        self.intent_patterns = {
            QueryIntent.HOW_TO: [
                r"\bhow\s+(do|to|can|should)\b",
                r"\bsteps?\s+to\b",
                r"\bguide\b",
                r"\btutorial\b",
            ],
            QueryIntent.DEBUG: [
                r"\berror\b",
                r"\bexception\b",
                r"\bfail(ing|ed|s)?\b",
                r"\bcrash(ing|ed)?\b",
                r"\bnot\s+work(ing)?\b",
                r"\bproblem\b",
                r"\bissue\b",
                r"\bbug\b",
                r"\bsegfault\b",
                r"\bcore\s+dump\b",
            ],
            QueryIntent.CODE_EXAMPLE: [
                r"\bexample\b",
                r"\bcode\s+snippet\b",
                r"\bsample\s+code\b",
                r"\bdemonstrat(e|ion)\b",
                r"\bshow\s+me\b",
            ],
            QueryIntent.DOCUMENTATION: [
                r"\bapi\s+reference\b",
                r"\bdocumentation\b",
                r"\bspecification\b",
                r"\bparameters?\b",
                r"\breturn\s+type\b",
                r"\bfunction\s+signature\b",
                r"\bclass\s+definition\b",
            ],
            QueryIntent.MAIL_HISTORY: [
                r"\bdiscussion\b",
                r"\bthread\b",
                r"\bemail\b",
                r"\bmail(ing\s+list)?\b",
                r"\bconversation\b",
                r"\bwas\s+(discussed|mentioned)\b",
            ],
            QueryIntent.CONCEPT: [
                r"\bwhat\s+is\b",
                r"\bexplain\b",
                r"\bdefinition\b",
                r"\bmeaning\b",
                r"\bunderstand\b",
                r"\bconcept\b",
            ],
            QueryIntent.COMPARISON: [
                r"\bcompare\b",
                r"\bdifference\s+between\b",
                r"\bvs\.?\b",
                r"\bversus\b",
                r"\b(better|worse)\s+than\b",
                r"\bwhich\s+to\s+use\b",
            ],
        }

    def classify(self, query: str, use_llm: bool = False) -> Tuple[QueryIntent, float]:
        """
        Classify query intent.

        Returns:
            (intent, confidence): Intent type and confidence score (0-1)
        """
        query_lower = query.lower()

        # Try LLM classification if available
        if use_llm and self.llm_client:
            try:
                return self._llm_classify(query)
            except Exception as e:
                self.logger.warning(f"⚠️ LLM classification failed: {e}, using rules")

        # Rule-based classification
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1

            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            return QueryIntent.UNKNOWN, 0.5

        # Get intent with highest score
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]

        # Normalize confidence
        total_patterns = len(self.intent_patterns[best_intent])
        confidence = min(max_score / total_patterns, 1.0)

        return best_intent, confidence

    def _llm_classify(self, query: str) -> Tuple[QueryIntent, float]:
        """Use LLM to classify intent."""
        prompt = f"""Classify the intent of this query into one of these categories:
- DOCUMENTATION: Looking for API docs, reference, specifications
- HOW_TO: How to accomplish a task, guide, tutorial
- DEBUG: Troubleshooting errors, bugs, crashes
- CODE_EXAMPLE: Looking for code examples, samples
- MAIL_HISTORY: Searching discussions, emails, threads
- CONCEPT: Understanding what something is, explanations
- COMPARISON: Comparing options, differences

Query: {query}

Classification (respond with just the category name):"""

        if hasattr(self.llm_client, "generate"):
            response = self.llm_client.generate(prompt, max_tokens=20).strip().upper()

            # Parse response
            for intent in QueryIntent:
                if intent.name in response:
                    return intent, 0.8  # High confidence from LLM

        # Fallback
        return QueryIntent.UNKNOWN, 0.3

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from query (libraries, functions, classes, etc.).
        Useful for targeted retrieval.
        """
        entities = {
            "libraries": [],
            "functions": [],
            "classes": [],
            "namespaces": [],
            "keywords": [],
        }

        # C++/Boost specific patterns
        # Namespaces: boost::asio::
        namespace_pattern = r"\b(boost|std)::[a-zA-Z_][a-zA-Z0-9_:]*"
        namespaces = re.findall(namespace_pattern, query)
        entities["namespaces"] = list(set(namespaces))

        # Functions: word followed by ()
        function_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        functions = re.findall(function_pattern, query)
        entities["functions"] = list(set(functions))

        # Classes/types (capitalized words)
        class_pattern = r"\b([A-Z][a-zA-Z0-9_]*)\b"
        classes = re.findall(class_pattern, query)
        entities["classes"] = list(set(classes))

        # Boost libraries
        boost_libs = [
            "asio",
            "thread",
            "beast",
            "filesystem",
            "regex",
            "spirit",
            "graph",
            "multi_index",
            "container",
            "algorithm",
        ]
        for lib in boost_libs:
            if lib in query.lower():
                entities["libraries"].append(lib)

        # Technical keywords
        keywords = [
            "async",
            "sync",
            "thread",
            "mutex",
            "lock",
            "shared_ptr",
            "unique_ptr",
            "template",
            "const",
            "virtual",
            "override",
        ]
        for keyword in keywords:
            if keyword in query.lower():
                entities["keywords"].append(keyword)

        return entities


class IntentRouter:
    """Route queries to appropriate retrieval strategies based on intent."""

    def __init__(self, classifier: Optional[QueryIntentClassifier] = None):
        self.logger = logger.bind(name="IntentRouter")
        self.classifier = classifier or QueryIntentClassifier()

        # Intent-specific configurations
        self.intent_configs = {
            QueryIntent.DOCUMENTATION: {
                "weights": {"semantic": 0.3, "bm25": 0.4, "graph": 0.3},
                "top_k": 10,
                "rerank": True,
                "threshold": 0.75,
                "preferred_sources": ["official_docs", "api_reference"],
                "graph_depth": 1,
            },
            QueryIntent.HOW_TO: {
                "weights": {"semantic": 0.5, "bm25": 0.2, "graph": 0.3},
                "top_k": 15,
                "rerank": True,
                "threshold": 0.65,
                "preferred_sources": ["tutorials", "guides", "examples"],
                "graph_depth": 2,
                "enable_multi_query": True,
            },
            QueryIntent.DEBUG: {
                "weights": {"semantic": 0.4, "bm25": 0.3, "graph": 0.3},
                "top_k": 20,
                "rerank": True,
                "threshold": 0.60,
                "preferred_sources": ["mail_archive", "stackoverflow", "issues"],
                "graph_depth": 2,
                "include_similar_errors": True,
            },
            QueryIntent.CODE_EXAMPLE: {
                "weights": {"semantic": 0.4, "bm25": 0.4, "graph": 0.2},
                "top_k": 10,
                "rerank": True,
                "threshold": 0.70,
                "preferred_sources": ["examples", "code_samples"],
                "filter_code_blocks": True,
            },
            QueryIntent.MAIL_HISTORY: {
                "weights": {"semantic": 0.3, "bm25": 0.2, "graph": 0.5},
                "top_k": 15,
                "rerank": True,
                "threshold": 0.65,
                "preferred_sources": ["mail_archive"],
                "graph_depth": 3,
                "use_thread_context": True,
                "sender_authority": True,
            },
            QueryIntent.CONCEPT: {
                "weights": {"semantic": 0.6, "bm25": 0.2, "graph": 0.2},
                "top_k": 8,
                "rerank": True,
                "threshold": 0.70,
                "preferred_sources": ["documentation", "tutorials"],
                "graph_depth": 1,
            },
            QueryIntent.COMPARISON: {
                "weights": {"semantic": 0.5, "bm25": 0.3, "graph": 0.2},
                "top_k": 12,
                "rerank": True,
                "threshold": 0.70,
                "diversity": True,
                "max_per_source": 2,
            },
            QueryIntent.UNKNOWN: {
                "weights": {"semantic": 0.5, "bm25": 0.25, "graph": 0.25},
                "top_k": 10,
                "rerank": True,
                "threshold": 0.70,
            },
        }

    def route(self, query: str) -> Dict[str, Any]:
        """
        Route query and return configuration for retrieval.

        Returns:
            Dict with retrieval configuration based on intent
        """
        # Classify intent
        intent, confidence = self.classifier.classify(query)

        # Extract entities
        entities = self.classifier.extract_entities(query)

        # Get configuration for this intent
        config = self.intent_configs.get(
            intent, self.intent_configs[QueryIntent.UNKNOWN]
        )

        # Add intent and confidence to config
        routing_config = config.copy()
        routing_config["intent"] = intent.value
        routing_config["intent_confidence"] = confidence
        routing_config["entities"] = entities

        self.logger.info(
            f"🎯 Query routed to {intent.value} (confidence: {confidence:.2f})"
        )

        return routing_config

    def apply_config(self, retriever, config: Dict[str, Any]):
        """Apply routing configuration to a retriever."""
        # Set weights if supported
        if hasattr(retriever, "set_weights"):
            retriever.set_weights(config.get("weights", {}))

        # Set threshold
        if hasattr(retriever, "set_threshold"):
            retriever.set_threshold(config.get("threshold", 0.7))

        # Configure other parameters
        if hasattr(retriever, "configure"):
            retriever.configure(config)

        return retriever

    def get_config(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get configuration for a specific intent."""
        return self.intent_configs.get(intent, self.intent_configs[QueryIntent.UNKNOWN])


class QueryAnalyzer:
    """Analyze query characteristics for optimization."""

    def __init__(self):
        self.logger = logger.bind(name="QueryAnalyzer")

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine optimal retrieval strategy.

        Returns:
            Dict with analysis results
        """
        analysis = {
            "length": len(query.split()),
            "has_code": self._has_code(query),
            "has_technical_terms": self._has_technical_terms(query),
            "is_question": query.strip().endswith("?"),
            "complexity": self._estimate_complexity(query),
            "language_hints": self._detect_language_hints(query),
        }

        return analysis

    def _has_code(self, query: str) -> bool:
        """Check if query contains code snippets."""
        code_indicators = ["()", "::", "->", ".", "template<", "class ", "struct "]
        return any(indicator in query for indicator in code_indicators)

    def _has_technical_terms(self, query: str) -> bool:
        """Check if query contains technical/domain terms."""
        technical_terms = [
            "async",
            "mutex",
            "thread",
            "buffer",
            "socket",
            "template",
            "iterator",
            "container",
            "algorithm",
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in technical_terms)

    def _estimate_complexity(self, query: str) -> str:
        """Estimate query complexity."""
        word_count = len(query.split())

        has_code = self._has_code(query)
        has_technical = self._has_technical_terms(query)

        if word_count <= 3 and not has_code:
            return "simple"
        elif word_count > 15 or (has_code and has_technical):
            return "complex"
        else:
            return "medium"

    def _detect_language_hints(self, query: str) -> List[str]:
        """Detect programming language hints in query."""
        hints = []

        if "c++" in query.lower() or "cpp" in query.lower():
            hints.append("cpp")
        if "python" in query.lower():
            hints.append("python")
        if "boost" in query.lower():
            hints.append("boost")

        return hints
