"""
Hybrid retrieval system combining semantic search, BM25, and Graph RAG.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import networkx as nx
from sentence_transformers import SentenceTransformer
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_config, RetrievalResult, get_model_nick_name



class DocumentGraphRAG:
    """Document-based Graph-based RAG for capturing relationships between concepts."""
    
    def __init__(self, language: str = None, graph_filename: str = None, embedder: SentenceTransformer = None):
        self.logger = logger.bind(name="DocumentGraphRAG")
        if language is not None:
            self.language = language
        
        self.embedder = embedder
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.relationship_types = {
            'function_calls': 'calls',
            'inherits_from': 'inherits',
            'uses': 'uses',
            'implements': 'implements',
            'example_of': 'example',
            'related_to': 'related'
        }
        if graph_filename is not None:
            self.graph_filename = graph_filename
        else:
            self.graph_filename = get_config("data.processed_data.graph_filename", "knowledge_graph.pkl")
    
    def extract_entities(self, text: str, chunk_type: str) -> List[str]:
        """Extract entities from text based on chunk type."""
        entities = []
        
        if chunk_type == 'api_reference':
            # Extract function names, class names
            func_pattern = r'(\w+)\s*\([^)]*\)'
            class_pattern = r'(?:class|struct)\s+(\w+)'
            entities.extend(re.findall(func_pattern, text))
            entities.extend(re.findall(class_pattern, text))
        
        elif chunk_type == 'code':
            # Extract variable names, function calls
            var_pattern = r'\b(\w+)\s*[=;]'
            call_pattern = r'(\w+)\s*\('
            entities.extend(re.findall(var_pattern, text))
            entities.extend(re.findall(call_pattern, text))
        
        elif chunk_type == 'header':
            # Extract section titles
            header_pattern = r'^#{1,6}\s+(.+)$'
            entities.extend(re.findall(header_pattern, text, re.MULTILINE))
        
        return list(set(entities))  # Remove duplicates
    
    def extract_relationships(self, text: str, chunk_type: str) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities."""
        relationships = []
        
        if chunk_type == 'code':
            # Function calls
            call_pattern = r'(\w+)\s*\.\s*(\w+)\s*\('
            for match in re.finditer(call_pattern, text):
                relationships.append((match.group(1), match.group(2), 'calls'))
            
            # Include relationships
            include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
            for match in re.finditer(include_pattern, text):
                relationships.append(('current_file', match.group(1), 'includes'))
        
        return relationships
    
    def reset_model(self, embedder: SentenceTransformer):
        """Reset the model."""
        self.embedder = embedder
    
    def update_graph(self, chunks: List[Dict[str, Any]], graph_file_path: str=None):
        """Update the knowledge graph."""
        self.logger.info("Updating knowledge graph...")
        if self.graph.number_of_nodes() == 0:
            self.load_graph(graph_file_path)
        self.build_graph(chunks, graph_file_path)
        self.save_graph(graph_file_path)
        self.logger.info(f"Knowledge graph updated with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def build_graph(self, chunks: List[Dict[str, Any]]=None):
        """Build knowledge graph from chunks."""
        self.logger.info("Building knowledge graph...")
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            text = chunk['text']
            metadata = chunk['metadata']
            chunk_type = metadata.get('chunk_type', 'text')
            
            # Add chunk as node
            self.graph.add_node(chunk_id, 
                              text=text, 
                              **metadata)
            
            # Extract and add entities
            entities = self.extract_entities(text, chunk_type)
            for entity in entities:
                if entity not in self.graph:
                    self.graph.add_node(entity, node_type='entity')
                self.graph.add_edge(chunk_id, entity, relationship='contains')
            
            # Extract and add relationships
            relationships = self.extract_relationships(text, chunk_type)
            for source, target, rel_type in relationships:
                if source in self.graph and target in self.graph:
                    self.graph.add_edge(source, target, relationship=rel_type)
        
        # Create entity embeddings
        entities = [node for node, data in self.graph.nodes(data=True) 
                   if data.get('node_type') == 'entity']
        if entities:
            entity_embeddings = self.embedder.encode(entities)
            for entity, embedding in zip(entities, entity_embeddings):
                self.entity_embeddings[entity] = embedding
        
        self.logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def save_graph(self, filepath: str=None) -> bool:
        """Save the knowledge graph and entity embeddings to disk."""
        try:
            import pickle
            import os
            
            # Create directory if it doesn't exist
            if filepath is None:
                filepath = get_config("data.processed_data.graph_file_path", "data/processed/graph") 
                filepath = f"{filepath}/{self.language}"
                filepath = f"{filepath}/{get_model_nick_name(self.embedder.model_card_data.base_model)}"
                filepath = f"{filepath}/{self.graph_filename}"
                
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Prepare data for saving
            graph_data = {
                'graph': self.graph,
                'entity_embeddings': self.entity_embeddings,
                'relationship_types': self.relationship_types
            }
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(graph_data, f)
            
            self.logger.info(f"Knowledge graph saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {e}")
            return False
    
    def load_graph(self, filepath: str=None) -> bool:
        """Load the knowledge graph and entity embeddings from disk."""
        try:
            import pickle
            import os
            if filepath is None:
                filepath = get_config("data.processed_data.graph_save_path", "data/processed/graph_cache") 
                filepath = f"{filepath}/{get_model_nick_name(self.embedder.model_card_data.base_model)}/{self.language}"
                filepath = f"{filepath}/{self.graph_filename}"
                
            if not os.path.exists(filepath):
                self.logger.warning(f"Graph file not found: {filepath}")
                return False
            
            # Load from file
            with open(filepath, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Restore graph data
            self.graph = graph_data['graph']
            self.entity_embeddings = graph_data['entity_embeddings']
            self.relationship_types = graph_data.get('relationship_types', self.relationship_types)
            
            self.logger.info(f"Knowledge graph loaded from {filepath}")
            self.logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")
            return False
    
    def search_graph(self, query: str, embedder: SentenceTransformer, top_k: int = 5) -> List[RetrievalResult]:
        """Search the knowledge graph."""
        if not self.entity_embeddings:
            return []
        
        # Get query embedding
        query_embedding = embedder.encode([query])[0]
        
        # Find similar entities
        entity_scores = []
        for entity, embedding in self.entity_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            entity_scores.append((entity, similarity))
        
        # Sort by similarity
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get chunks connected to top entities
        results = []
        seen_chunks = set()
        
        for entity, score in entity_scores[:top_k * 2]:
            # Find chunks connected to this entity
            for chunk_id in self.graph.predecessors(entity):
                if chunk_id in seen_chunks:
                    continue
                
                chunk_data = self.graph.nodes[chunk_id]
                if 'text' in chunk_data:
                    results.append(RetrievalResult(
                        text=chunk_data['text'],
                        score=score * 0.8,  # Weight graph results lower
                        metadata=chunk_data,
                        retrieval_method='graph',
                        source_file=chunk_data.get('source_file', ''),
                        source_type=chunk_data.get('source_file', '').split('.')[-1]
                    ))
                    seen_chunks.add(chunk_id)
        
        return results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the document graph."""
        try:
            stats = {
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "entity_embeddings_count": len(self.entity_embeddings),
                "language": getattr(self, 'language', 'unknown'),
                "graph_filename": getattr(self, 'graph_filename', 'unknown'),
                "status": "initialized" if self.graph.number_of_nodes() > 0 else "empty"
            }
            
            # Add entity statistics if available
            if self.entity_embeddings:
                stats["embedding_dimension"] = len(next(iter(self.entity_embeddings.values()))) if self.entity_embeddings else 0
                stats["entity_types"] = len(set(entity.split('_')[0] for entity in self.entity_embeddings.keys()))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting document graph stats: {e}")
            return {
                "graph_nodes": 0,
                "graph_edges": 0,
                "entity_embeddings_count": 0,
                "status": "error",
                "error": str(e)
            }

def main():
    """Main function for testing hybrid retrieval."""
    # This would be used for testing the hybrid retrieval system
    pass


if __name__ == "__main__":
    main()
