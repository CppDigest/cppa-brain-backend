"""
Simple RAG Evaluation System
A lightweight evaluation module for RAG system performance.
"""

import json
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag.improved_rag_system import RAGSystem, SearchRequest, SearchMethod, SearchScope


class SimpleRAGEvaluator:
    """Simple RAG evaluation system."""
    
    def __init__(self, rag_system: RAGSystem = None):
        """Initialize the evaluator."""
        self.logger = logger.bind(name="SimpleRAGEvaluator")
        
        # Initialize RAG system
        self.rag_system = rag_system
        
        # Initialize similarity model
        try:
            self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info("Similarity model loaded")
        except Exception as e:
            self.logger.error(f"Failed to load similarity model: {e}")
            self.similarity_model = None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if self.similarity_model is None:
            return 0.5
        
        try:
            embeddings = self.similarity_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.5
    
    def evaluate_question(self, question: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate a single question."""
        start_time = time.time()
        
        if self.rag_system is None:
            return {
                "question": question,
                "ground_truth": ground_truth,
                "rag_answer": "RAG system not available",
                "similarity": 0.0,
                "time": 0.0,
                "error": "RAG system not initialized"
            }
        
        try:
            # Get search results
            search_request = SearchRequest(
                query=question,
                method=SearchMethod.HYBRID,
                scope=SearchScope.BOTH,
                max_results=5
            )
            
            search_response = self.rag_system.orchestrator.search(search_request)
            
            # Create simple answer from search results
            rag_answer = self._create_simple_answer(search_response.results)
            
            # Calculate similarity
            similarity = self.calculate_similarity(ground_truth, rag_answer)
            
            evaluation_time = time.time() - start_time
            
            return {
                "question": question,
                "ground_truth": ground_truth,
                "rag_answer": rag_answer,
                "similarity": similarity,
                "time": evaluation_time,
                "num_results": len(search_response.results)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating question: {e}")
            return {
                "question": question,
                "ground_truth": ground_truth,
                "rag_answer": f"Error: {str(e)}",
                "similarity": 0.0,
                "time": time.time() - start_time,
                "error": str(e)
            }
    
    def _create_simple_answer(self, results: List[Any]) -> str:
        """Create a simple answer from search results."""
        if not results:
            return "No relevant information found."
        
        # Combine top results
        answer_parts = []
        for i, result in enumerate(results[:3]):  # Use top 3 results
            answer_parts.append(f"{result.text[:200]}...")
        
        return " ".join(answer_parts)
    
    def evaluate_dataset(self, dataset_path: str=None, max_questions: int = 5) -> Dict[str, Any]:
        """Evaluate a validation dataset."""
        if dataset_path is None:
            dataset_path = "data/validation_dataset.json"
        self.logger.info(f"Evaluating dataset: {dataset_path}")
        
        # Load dataset
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'validation_data' in data:
                questions = data['validation_data']
            else:
                questions = data
                
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return {"error": f"Failed to load dataset: {e}"}
        
        # Limit questions if specified
        if max_questions:
            questions = questions[:max_questions]
        
        # Evaluate each question
        results = []
        similarities = []
        times = []
        
        for i, item in enumerate(questions):
            self.logger.info(f"Evaluating question {i+1}/{len(questions)}")
            
            question = item.get('question', '')
            ground_truth = item.get('answer', '')
            
            if not question or not ground_truth:
                continue
            
            result = self.evaluate_question(question, ground_truth)
            results.append(result)
            
            if 'similarity' in result:
                similarities.append(result['similarity'])
            if 'time' in result:
                times.append(result['time'])
        
        # Calculate summary metrics
        avg_similarity = np.mean(similarities) if similarities else 0.0
        avg_time = np.mean(times) if times else 0.0
        good_answers = sum(1 for s in similarities if s > 0.7)
        accuracy = good_answers / len(similarities) if similarities else 0.0
        
        metrics = {
            "total_questions": len(results),
            "average_similarity": avg_similarity,
            "average_time": avg_time,
            "accuracy": accuracy,
            "good_answers": good_answers,
            "results": results
        }
        self.print_summary(metrics)
        return metrics
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("RAG EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Average Similarity: {metrics['average_similarity']:.3f}")
        print(f"Average Time: {metrics['average_time']:.3f}s")
        print(f"Accuracy (>0.7): {metrics['accuracy']:.3f}")
        print(f"Good Answers: {metrics['good_answers']}/{metrics['total_questions']}")
        print("="*50)


def main():
    """Main function for testing."""
    evaluator = SimpleRAGEvaluator()
    
    # Test with validation dataset
    dataset_path = "data/validation_dataset.json"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(dataset_path, max_questions=5)
    
    # Print results
    evaluator.print_summary(metrics)
    
    # Save results
    with open("simple_evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: simple_evaluation_results.json")


if __name__ == "__main__":
    main()
