"""
LLM-as-a-Judge evaluation system with groundedness metrics.
Automatically evaluates answer faithfulness, relevance, and groundedness.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_config


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    faithfulness_score: float
    relevance_score: float
    groundedness_score: float
    completeness_score: float
    accuracy_score: float
    overall_score: float
    detailed_feedback: Dict[str, Any]
    issues_found: List[str]
    strengths: List[str]


@dataclass
class GroundednessAnalysis:
    """Analysis of how well the answer is grounded in sources."""
    grounded_claims: List[str]
    ungrounded_claims: List[str]
    grounding_confidence: float
    source_coverage: float
    citation_quality: float


class LLMJudge:
    """LLM-as-a-Judge evaluation system."""
    
    def __init__(self):
        self.logger = logger.bind(name="LLMJudge")
        
        # Initialize OpenAI client
        self.client = None
        self._initialize_client()
        
        # Initialize sentence transformer for similarity
        self.similarity_model = None
        self._initialize_similarity_model()
        
        # Configuration
        self.judge_model = get_config("rag.evaluation.judge_model", "gpt-4")
        self.groundedness_threshold = get_config("rag.evaluation.groundedness_threshold", 0.8)
        self.faithfulness_threshold = get_config("rag.evaluation.faithfulness_threshold", 0.7)
        
        # Load evaluation prompts
        self.evaluation_prompts = self._load_evaluation_prompts()
        
        self.logger.info("LLMJudge initialized")
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                api_key = get_config("rag.llm.api_key", None)
            
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.logger.info("OpenAI client initialized for evaluation")
            else:
                self.logger.warning("No OpenAI API key found for evaluation")
                self.client = None
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            self.client = None
    
    def _initialize_similarity_model(self):
        """Initialize sentence transformer for similarity calculations."""
        try:
            model_name = get_config("rag.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            self.similarity_model = SentenceTransformer(model_name)
            self.logger.info("Similarity model loaded for evaluation")
        except Exception as e:
            self.logger.error(f"Error loading similarity model: {e}")
            self.similarity_model = None
    
    def _load_evaluation_prompts(self) -> Dict[str, str]:
        """Load evaluation prompts for different metrics."""
        prompts = {
            'faithfulness': """You are an expert evaluator for technical documentation answers. Your task is to evaluate the faithfulness of an answer to its source context.

FAITHFULNESS CRITERIA:
1. Does the answer only use information present in the context?
2. Are all claims in the answer supported by the context?
3. Are there any hallucinations or made-up information?
4. Are function names, parameters, and code examples accurate?
5. Is the technical information correct according to the context?

EVALUATION SCALE:
- 1.0: Perfectly faithful, all information is in context
- 0.8-0.9: Mostly faithful, minor issues
- 0.6-0.7: Somewhat faithful, some unsupported claims
- 0.4-0.5: Not very faithful, significant issues
- 0.0-0.3: Not faithful, major hallucinations

CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}

Please provide:
1. Faithfulness score (0.0 to 1.0)
2. Specific issues found (if any)
3. Strengths of the answer
4. Detailed reasoning

RESPONSE FORMAT:
Score: [0.0-1.0]
Issues: [list of specific issues]
Strengths: [list of strengths]
Reasoning: [detailed explanation]""",

            'relevance': """You are an expert evaluator for technical documentation answers. Your task is to evaluate the relevance of an answer to the user's question.

RELEVANCE CRITERIA:
1. Does the answer directly address the user's question?
2. Is the level of detail appropriate for the question?
3. Are all parts of the answer relevant to the question?
4. Does the answer provide what the user is looking for?
5. Is the answer focused and not overly broad or narrow?

EVALUATION SCALE:
- 1.0: Perfectly relevant, directly answers the question
- 0.8-0.9: Highly relevant, minor gaps
- 0.6-0.7: Somewhat relevant, some irrelevant parts
- 0.4-0.5: Not very relevant, significant gaps
- 0.0-0.3: Not relevant, doesn't answer the question

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

Please provide:
1. Relevance score (0.0 to 1.0)
2. Specific relevance issues (if any)
3. Strengths of the answer
4. Detailed reasoning

RESPONSE FORMAT:
Score: [0.0-1.0]
Issues: [list of specific issues]
Strengths: [list of strengths]
Reasoning: [detailed explanation]""",

            'completeness': """You are an expert evaluator for technical documentation answers. Your task is to evaluate the completeness of an answer.

COMPLETENESS CRITERIA:
1. Does the answer fully address all parts of the question?
2. Are important details included?
3. Should code examples be more complete?
4. Are there related concepts that should be mentioned?
5. Is the answer at an appropriate level of detail?

EVALUATION SCALE:
- 1.0: Complete, covers all aspects
- 0.8-0.9: Mostly complete, minor gaps
- 0.6-0.7: Somewhat complete, some missing details
- 0.4-0.5: Not very complete, significant gaps
- 0.0-0.3: Incomplete, major missing information

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

Please provide:
1. Completeness score (0.0 to 1.0)
2. Missing information (if any)
3. Strengths of the answer
4. Detailed reasoning

RESPONSE FORMAT:
Score: [0.0-1.0]
Issues: [list of missing information]
Strengths: [list of strengths]
Reasoning: [detailed explanation]""",

            'accuracy': """You are an expert evaluator for Boost.Asio documentation answers. Your task is to evaluate the technical accuracy of an answer.

ACCURACY CRITERIA:
1. Are Boost.Asio function names and signatures correct?
2. Are the code examples syntactically correct?
3. Are the concepts explained accurately?
4. Are best practices mentioned correctly?
5. Are there any technical errors?

EVALUATION SCALE:
- 1.0: Technically accurate, no errors
- 0.8-0.9: Mostly accurate, minor issues
- 0.6-0.7: Somewhat accurate, some errors
- 0.4-0.5: Not very accurate, significant errors
- 0.0-0.3: Inaccurate, major technical errors

ANSWER TO EVALUATE:
{answer}

Please provide:
1. Accuracy score (0.0 to 1.0)
2. Technical errors found (if any)
3. Strengths of the answer
4. Detailed reasoning

RESPONSE FORMAT:
Score: [0.0-1.0]
Issues: [list of technical errors]
Strengths: [list of strengths]
Reasoning: [detailed explanation]"""
        }
        
        return prompts
    
    def _parse_evaluation_response(self, response: str) -> Tuple[float, List[str], List[str], str]:
        """Parse evaluation response from LLM."""
        lines = response.strip().split('\n')
        
        score = 0.5
        issues = []
        strengths = []
        reasoning = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Score:'):
                score_match = re.search(r'(\d+\.?\d*)', line)
                if score_match:
                    score = float(score_match.group(1))
                    if score > 1.0:
                        score /= 100.0  # Convert percentage to decimal
            elif line.startswith('Issues:'):
                current_section = 'issues'
            elif line.startswith('Strengths:'):
                current_section = 'strengths'
            elif line.startswith('Reasoning:'):
                current_section = 'reasoning'
            elif line.startswith('Missing information:'):
                current_section = 'issues'
            elif line.startswith('Technical errors:'):
                current_section = 'issues'
            elif line.startswith('Relevance issues:'):
                current_section = 'issues'
            elif line.startswith('Faithfulness issues:'):
                current_section = 'issues'
            elif line.startswith('-') or line.startswith('*'):
                content = line[1:].strip()
                if current_section == 'issues':
                    issues.append(content)
                elif current_section == 'strengths':
                    strengths.append(content)
            elif current_section == 'reasoning':
                reasoning += line + " "
        
        return score, issues, strengths, reasoning.strip()
    
    def _evaluate_metric(self, metric_type: str, question: str, answer: str, context: str) -> Tuple[float, List[str], List[str], str]:
        """Evaluate a specific metric using LLM."""
        if self.client is None:
            # Mock evaluation when LLM is not available
            return 0.7, [], ["Mock evaluation - LLM not available"], "Mock evaluation performed"
        
        try:
            prompt_template = self.evaluation_prompts.get(metric_type, self.evaluation_prompts['faithfulness'])
            prompt = prompt_template.format(
                question=question,
                answer=answer,
                context=context
            )
            
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            evaluation_response = response.choices[0].message.content
            return self._parse_evaluation_response(evaluation_response)
            
        except Exception as e:
            self.logger.error(f"Error evaluating {metric_type}: {e}")
            return 0.0, [f"Evaluation error: {str(e)}"], [], "Error during evaluation"
    
    def _analyze_groundedness(self, answer: str, context: str) -> GroundednessAnalysis:
        """Analyze how well the answer is grounded in the context."""
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        grounded_claims = []
        ungrounded_claims = []
        
        for claim in claims:
            if self._is_claim_grounded(claim, context):
                grounded_claims.append(claim)
            else:
                ungrounded_claims.append(claim)
        
        # Calculate metrics
        total_claims = len(claims)
        grounding_confidence = len(grounded_claims) / total_claims if total_claims > 0 else 0.0
        
        # Calculate source coverage
        source_coverage = self._calculate_source_coverage(answer, context)
        
        # Calculate citation quality
        citation_quality = self._calculate_citation_quality(answer, context)
        
        return GroundednessAnalysis(
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
            grounding_confidence=grounding_confidence,
            source_coverage=source_coverage,
            citation_quality=citation_quality
        )
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract claims from text."""
        # Simple claim extraction - can be improved
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.startswith(('According to', 'Based on', 'As shown')):
                claims.append(sentence)
        
        return claims
    
    def _is_claim_grounded(self, claim: str, context: str) -> bool:
        """Check if a claim is grounded in the context."""
        if self.similarity_model is None:
            # Simple keyword-based grounding check
            claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
            context_words = set(re.findall(r'\b\w+\b', context.lower()))
            
            # Check if significant portion of claim words are in context
            overlap = len(claim_words.intersection(context_words))
            return overlap / len(claim_words) > 0.5 if claim_words else False
        
        try:
            # Use semantic similarity
            embeddings = self.similarity_model.encode([claim, context])
            similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            return similarity > 0.7
        except Exception as e:
            self.logger.error(f"Error checking claim grounding: {e}")
            return False
    
    def _calculate_source_coverage(self, answer: str, context: str) -> float:
        """Calculate how well the answer covers the available sources."""
        # Simple heuristic - can be improved
        answer_length = len(answer)
        context_length = len(context)
        
        if context_length == 0:
            return 0.0
        
        # Check if answer references specific parts of context
        context_references = len(re.findall(r'\[Context \d+\]', answer))
        return min(1.0, context_references / 3.0)  # Normalize to 0-1
    
    def _calculate_citation_quality(self, answer: str, context: str) -> float:
        """Calculate the quality of citations in the answer."""
        # Check for explicit citations
        citation_patterns = [
            r'\[Context \d+\]',
            r'according to',
            r'based on',
            r'as shown in',
            r'the documentation',
            r'the example'
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, answer, re.IGNORECASE))
        
        # Normalize based on answer length
        answer_sentences = len(re.split(r'[.!?]+', answer))
        return min(1.0, citation_count / max(1, answer_sentences))
    
    def evaluate_answer(self, question: str, answer: str, context: str) -> EvaluationMetrics:
        """
        Comprehensive evaluation of an answer.
        
        Args:
            question: Original user question
            answer: Generated answer
            context: Source context used
            
        Returns:
            EvaluationMetrics with all scores and feedback
        """
        try:
            self.logger.info("Starting comprehensive answer evaluation")
            
            # Evaluate different metrics
            faithfulness_score, faithfulness_issues, faithfulness_strengths, faithfulness_reasoning = self._evaluate_metric(
                'faithfulness', question, answer, context
            )
            
            relevance_score, relevance_issues, relevance_strengths, relevance_reasoning = self._evaluate_metric(
                'relevance', question, answer, context
            )
            
            completeness_score, completeness_issues, completeness_strengths, completeness_reasoning = self._evaluate_metric(
                'completeness', question, answer, context
            )
            
            accuracy_score, accuracy_issues, accuracy_strengths, accuracy_reasoning = self._evaluate_metric(
                'accuracy', question, answer, context
            )
            
            # Analyze groundedness
            groundedness_analysis = self._analyze_groundedness(answer, context)
            groundedness_score = groundedness_analysis.grounding_confidence
            
            # Calculate overall score
            overall_score = (
                faithfulness_score * 0.3 +
                relevance_score * 0.25 +
                completeness_score * 0.2 +
                accuracy_score * 0.15 +
                groundedness_score * 0.1
            )
            
            # Combine all issues and strengths
            all_issues = faithfulness_issues + relevance_issues + completeness_issues + accuracy_issues
            all_strengths = faithfulness_strengths + relevance_strengths + completeness_strengths + accuracy_strengths
            
            # Create detailed feedback
            detailed_feedback = {
                'faithfulness': {
                    'score': faithfulness_score,
                    'issues': faithfulness_issues,
                    'strengths': faithfulness_strengths,
                    'reasoning': faithfulness_reasoning
                },
                'relevance': {
                    'score': relevance_score,
                    'issues': relevance_issues,
                    'strengths': relevance_strengths,
                    'reasoning': relevance_reasoning
                },
                'completeness': {
                    'score': completeness_score,
                    'issues': completeness_issues,
                    'strengths': completeness_strengths,
                    'reasoning': completeness_reasoning
                },
                'accuracy': {
                    'score': accuracy_score,
                    'issues': accuracy_issues,
                    'strengths': accuracy_strengths,
                    'reasoning': accuracy_reasoning
                },
                'groundedness': {
                    'score': groundedness_score,
                    'grounded_claims': groundedness_analysis.grounded_claims,
                    'ungrounded_claims': groundedness_analysis.ungrounded_claims,
                    'source_coverage': groundedness_analysis.source_coverage,
                    'citation_quality': groundedness_analysis.citation_quality
                }
            }
            
            metrics = EvaluationMetrics(
                faithfulness_score=faithfulness_score,
                relevance_score=relevance_score,
                groundedness_score=groundedness_score,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                overall_score=overall_score,
                detailed_feedback=detailed_feedback,
                issues_found=all_issues,
                strengths=all_strengths
            )
            
            self.logger.info(f"Evaluation completed. Overall score: {overall_score:.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            return EvaluationMetrics(
                faithfulness_score=0.0,
                relevance_score=0.0,
                groundedness_score=0.0,
                completeness_score=0.0,
                accuracy_score=0.0,
                overall_score=0.0,
                detailed_feedback={'error': str(e)},
                issues_found=[f"Evaluation error: {str(e)}"],
                strengths=[]
            )
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the evaluation system."""
        return {
            'judge_model': self.judge_model,
            'groundedness_threshold': self.groundedness_threshold,
            'faithfulness_threshold': self.faithfulness_threshold,
            'available_metrics': list(self.evaluation_prompts.keys()),
            'client_available': self.client is not None,
            'similarity_model_available': self.similarity_model is not None
        }


def main():
    """Main function for testing evaluation system."""
    # This would be used for testing the evaluation system
    pass


if __name__ == "__main__":
    main()
