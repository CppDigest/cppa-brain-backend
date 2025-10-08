"""
RAG System Evaluation Module
Evaluates retrieval performance using validation dataset and local LLM similarity evaluation.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_config
from rag.improved_rag_system import ImprovedRAGSystem, SearchRequest, SearchMethod, SearchScope


@dataclass
class EvaluationResult:
    """Single evaluation result for a question-answer pair."""
    question: str
    ground_truth: str
    rag_answer: str
    similarity_score: float
    semantic_similarity: float
    llm_similarity_score: float
    retrieval_context: str
    evaluation_time: float
    retrieval_metadata: Dict[str, Any]


@dataclass
class EvaluationMetrics:
    """Overall evaluation metrics."""
    total_questions: int
    average_similarity: float
    average_semantic_similarity: float
    average_llm_similarity: float
    retrieval_accuracy: float
    response_time_avg: float
    detailed_results: List[EvaluationResult]
    evaluation_summary: Dict[str, Any]


class LocalLLMSimilarityEvaluator:
    """Local LLM-based similarity evaluator for RAG responses."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize local LLM evaluator."""
        self.logger = logger.bind(name="LocalLLMSimilarityEvaluator")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            self.logger.info(f"Local LLM loaded: {model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load local LLM: {e}")
            self.tokenizer = None
            self.model = None
    
    def evaluate_similarity(self, ground_truth: str, rag_answer: str) -> float:
        """Evaluate similarity between ground truth and RAG answer using local LLM."""
        if self.model is None or self.tokenizer is None:
            # Fallback to semantic similarity if LLM not available
            return self._semantic_similarity_fallback(ground_truth, rag_answer)
        
        try:
            # Create prompt for similarity evaluation
            prompt = f"""Compare the similarity between these two answers about Boost C++ libraries:

Ground Truth: {ground_truth}

RAG Answer: {rag_answer}

Rate the similarity on a scale of 0.0 to 1.0 where:
- 1.0 = Perfect match, identical meaning
- 0.8-0.9 = Very similar, minor differences
- 0.6-0.7 = Somewhat similar, some differences
- 0.4-0.5 = Different but related
- 0.0-0.3 = Very different or unrelated

Similarity score:"""
            
            # Tokenize and generate
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract similarity score from response
            similarity_score = self._extract_similarity_score(response)
            return similarity_score
            
        except Exception as e:
            self.logger.error(f"Error in LLM similarity evaluation: {e}")
            return self._semantic_similarity_fallback(ground_truth, rag_answer)
    
    def _extract_similarity_score(self, response: str) -> float:
        """Extract similarity score from LLM response."""
        try:
            # Look for numerical score in response
            import re
            score_match = re.search(r'(\d+\.?\d*)', response)
            if score_match:
                score = float(score_match.group(1))
                # Normalize to 0-1 range
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else score / 100.0
                return max(0.0, min(1.0, score))
        except:
            pass
        
        # Fallback: analyze response sentiment
        return self._analyze_response_sentiment(response)
    
    def _analyze_response_sentiment(self, response: str) -> float:
        """Analyze response sentiment to estimate similarity."""
        positive_words = ['similar', 'match', 'identical', 'same', 'correct', 'accurate']
        negative_words = ['different', 'unrelated', 'incorrect', 'wrong', 'mismatch']
        
        response_lower = response.lower()
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral if no clear indicators
        
        return positive_count / (positive_count + negative_count)
    
    def _semantic_similarity_fallback(self, ground_truth: str, rag_answer: str) -> float:
        """Fallback to semantic similarity when LLM is not available."""
        try:
            # Use sentence transformer for semantic similarity
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embeddings = model.encode([ground_truth, rag_answer])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Semantic similarity fallback failed: {e}")
            return 0.5


class RAGEvaluationModule:
    """Main evaluation module for RAG system performance."""
    
    def __init__(self, rag_system: Optional[ImprovedRAGSystem] = None):
        """Initialize evaluation module."""
        self.logger = logger.bind(name="RAGEvaluationModule")
        
        # Initialize RAG system
        if rag_system is None:
            self.rag_system = ImprovedRAGSystem()
            if not self.rag_system.initialize():
                self.logger.error("Failed to initialize RAG system")
                self.rag_system = None
        else:
            self.rag_system = rag_system
        
        # Initialize similarity evaluator
        self.similarity_evaluator = LocalLLMSimilarityEvaluator()
        
        # Initialize semantic similarity model
        try:
            self.semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info("Semantic similarity model loaded")
        except Exception as e:
            self.logger.error(f"Failed to load semantic model: {e}")
            self.semantic_model = None
    
    def load_validation_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load validation dataset from JSON file."""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'validation_data' in data:
                return data['validation_data']
            else:
                return data  # Assume direct list format
                
        except Exception as e:
            self.logger.error(f"Failed to load validation dataset: {e}")
            return []
    
    def evaluate_single_question(self, question: str, ground_truth: str) -> EvaluationResult:
        """Evaluate a single question-answer pair."""
        start_time = time.time()
        
        try:
            # Get RAG system response
            if self.rag_system is None:
                self.logger.error("RAG system not initialized")
                return EvaluationResult(
                    question=question,
                    ground_truth=ground_truth,
                    rag_answer="RAG system not available",
                    similarity_score=0.0,
                    semantic_similarity=0.0,
                    llm_similarity_score=0.0,
                    retrieval_context="",
                    evaluation_time=0.0,
                    retrieval_metadata={}
                )
            
            # Create search request
            search_request = SearchRequest(
                query=question,
                method=SearchMethod.HYBRID,
                scope=SearchScope.BOTH,
                max_results=5,
                use_reranker=True
            )
            
            # Get search results
            search_response = self.rag_system.search_orchestrator.search(search_request)
            
            # Extract context from search results
            context_parts = []
            for result in search_response.results:
                context_parts.append(result.text)
            
            retrieval_context = "\n\n".join(context_parts)
            
            # Generate answer using RAG system (simplified - you may need to adapt based on your RAG system)
            rag_answer = self._generate_rag_answer(question, retrieval_context)
            
            # Calculate similarity scores
            semantic_similarity = self._calculate_semantic_similarity(ground_truth, rag_answer)
            llm_similarity = self.similarity_evaluator.evaluate_similarity(ground_truth, rag_answer)
            
            # Overall similarity (weighted average)
            overall_similarity = (semantic_similarity * 0.4 + llm_similarity * 0.6)
            
            evaluation_time = time.time() - start_time
            
            return EvaluationResult(
                question=question,
                ground_truth=ground_truth,
                rag_answer=rag_answer,
                similarity_score=overall_similarity,
                semantic_similarity=semantic_similarity,
                llm_similarity_score=llm_similarity,
                retrieval_context=retrieval_context,
                evaluation_time=evaluation_time,
                retrieval_metadata={
                    "total_results": len(search_response.results),
                    "search_time": search_response.search_time,
                    "method_used": search_response.method_used
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating question: {e}")
            return EvaluationResult(
                question=question,
                ground_truth=ground_truth,
                rag_answer=f"Error: {str(e)}",
                similarity_score=0.0,
                semantic_similarity=0.0,
                llm_similarity_score=0.0,
                retrieval_context="",
                evaluation_time=time.time() - start_time,
                retrieval_metadata={"error": str(e)}
            )
    
    def _generate_rag_answer(self, question: str, context: str) -> str:
        """Generate answer using RAG system (simplified implementation)."""
        # This is a simplified implementation - you may need to adapt based on your RAG system
        # For now, return a basic response based on context
        if not context:
            return "No relevant context found for this question."
        
        # Simple answer generation (you may want to use your actual RAG generation pipeline)
        return f"Based on the available context: {context[:500]}..."
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if self.semantic_model is None:
            return 0.5  # Default similarity if model not available
        
        try:
            embeddings = self.semantic_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.5
    
    def evaluate_dataset(self, dataset_path: str, max_questions: Optional[int] = None) -> EvaluationMetrics:
        """Evaluate entire validation dataset."""
        self.logger.info(f"Starting evaluation of dataset: {dataset_path}")
        
        # Load validation dataset
        validation_data = self.load_validation_dataset(dataset_path)
        
        if not validation_data:
            self.logger.error("No validation data loaded")
            return EvaluationMetrics(
                total_questions=0,
                average_similarity=0.0,
                average_semantic_similarity=0.0,
                average_llm_similarity=0.0,
                retrieval_accuracy=0.0,
                response_time_avg=0.0,
                detailed_results=[],
                evaluation_summary={"error": "No validation data loaded"}
            )
        
        # Limit questions if specified
        if max_questions:
            validation_data = validation_data[:max_questions]
        
        self.logger.info(f"Evaluating {len(validation_data)} questions")
        
        # Evaluate each question
        detailed_results = []
        total_similarity = 0.0
        total_semantic_similarity = 0.0
        total_llm_similarity = 0.0
        total_time = 0.0
        
        for i, item in enumerate(validation_data):
            self.logger.info(f"Evaluating question {i+1}/{len(validation_data)}")
            
            question = item.get('question', '')
            ground_truth = item.get('answer', '')
            
            if not question or not ground_truth:
                self.logger.warning(f"Skipping invalid item: {item}")
                continue
            
            result = self.evaluate_single_question(question, ground_truth)
            detailed_results.append(result)
            
            total_similarity += result.similarity_score
            total_semantic_similarity += result.semantic_similarity
            total_llm_similarity += result.llm_similarity_score
            total_time += result.evaluation_time
        
        # Calculate metrics
        num_questions = len(detailed_results)
        if num_questions == 0:
            return EvaluationMetrics(
                total_questions=0,
                average_similarity=0.0,
                average_semantic_similarity=0.0,
                average_llm_similarity=0.0,
                retrieval_accuracy=0.0,
                response_time_avg=0.0,
                detailed_results=[],
                evaluation_summary={"error": "No valid questions evaluated"}
            )
        
        average_similarity = total_similarity / num_questions
        average_semantic_similarity = total_semantic_similarity / num_questions
        average_llm_similarity = total_llm_similarity / num_questions
        response_time_avg = total_time / num_questions
        
        # Calculate retrieval accuracy (questions with similarity > 0.7)
        high_similarity_count = sum(1 for r in detailed_results if r.similarity_score > 0.7)
        retrieval_accuracy = high_similarity_count / num_questions
        
        # Create evaluation summary
        evaluation_summary = {
            "total_questions": num_questions,
            "high_similarity_count": high_similarity_count,
            "retrieval_accuracy": retrieval_accuracy,
            "average_similarity": average_similarity,
            "average_semantic_similarity": average_semantic_similarity,
            "average_llm_similarity": average_llm_similarity,
            "average_response_time": response_time_avg,
            "similarity_distribution": {
                "excellent (>0.9)": sum(1 for r in detailed_results if r.similarity_score > 0.9),
                "good (0.7-0.9)": sum(1 for r in detailed_results if 0.7 <= r.similarity_score <= 0.9),
                "fair (0.5-0.7)": sum(1 for r in detailed_results if 0.5 <= r.similarity_score < 0.7),
                "poor (<0.5)": sum(1 for r in detailed_results if r.similarity_score < 0.5)
            }
        }
        
        metrics = EvaluationMetrics(
            total_questions=num_questions,
            average_similarity=average_similarity,
            average_semantic_similarity=average_semantic_similarity,
            average_llm_similarity=average_llm_similarity,
            retrieval_accuracy=retrieval_accuracy,
            response_time_avg=response_time_avg,
            detailed_results=detailed_results,
            evaluation_summary=evaluation_summary
        )
        
        self.logger.info(f"Evaluation completed. Average similarity: {average_similarity:.3f}")
        return metrics
    
    def save_evaluation_report(self, metrics: EvaluationMetrics, output_path: str):
        """Save evaluation report to file."""
        try:
            report = {
                "evaluation_metrics": asdict(metrics),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_module": "RAGEvaluationModule"
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Evaluation report saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save evaluation report: {e}")
    
    def print_evaluation_summary(self, metrics: EvaluationMetrics):
        """Print evaluation summary to console."""
        print("\n" + "="*60)
        print("RAG SYSTEM EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions Evaluated: {metrics.total_questions}")
        print(f"Average Similarity Score: {metrics.average_similarity:.3f}")
        print(f"Average Semantic Similarity: {metrics.average_semantic_similarity:.3f}")
        print(f"Average LLM Similarity: {metrics.average_llm_similarity:.3f}")
        print(f"Retrieval Accuracy (>0.7): {metrics.retrieval_accuracy:.3f}")
        print(f"Average Response Time: {metrics.response_time_avg:.3f}s")
        
        if metrics.evaluation_summary and "similarity_distribution" in metrics.evaluation_summary:
            dist = metrics.evaluation_summary["similarity_distribution"]
            print(f"\nSimilarity Distribution:")
            print(f"  Excellent (>0.9): {dist['excellent (>0.9)']}")
            print(f"  Good (0.7-0.9): {dist['good (0.7-0.9)']}")
            print(f"  Fair (0.5-0.7): {dist['fair (0.5-0.7)']}")
            print(f"  Poor (<0.5): {dist['poor (<0.5)']}")
        
        print("="*60)


def main():
    """Main function for running evaluation."""
    # Initialize evaluation module
    evaluator = RAGEvaluationModule()
    
    # Path to validation dataset
    dataset_path = "data/validation_dataset.json"
    
    if not os.path.exists(dataset_path):
        print(f"Validation dataset not found at: {dataset_path}")
        return
    
    # Run evaluation
    print("Starting RAG system evaluation...")
    metrics = evaluator.evaluate_dataset(dataset_path, max_questions=5)  # Limit for testing
    
    # Print summary
    evaluator.print_evaluation_summary(metrics)
    
    # Save detailed report
    output_path = "evaluation_report.json"
    evaluator.save_evaluation_report(metrics, output_path)
    
    print(f"\nDetailed evaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()
