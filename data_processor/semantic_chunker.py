"""
Semantic chunking module for document structure-aware text segmentation.
Handles Boost.Asio documentation with preservation of code blocks, headers, and structure.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_config
from data_processor.multiformat_processor import MultiFormatProcessor

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    source_file: str
    chunk_type: str  # 'code', 'header', 'text', 'example', 'api_reference'
    start_line: int
    end_line: int
    parent_section: Optional[str] = None
    function_name: Optional[str] = None
    example_type: Optional[str] = None
    importance_score: float = 1.0


class SemanticChunker:
    """Semantic chunker that preserves document structure and creates meaningful chunks."""
    
    def __init__(self, language: str = None, embedding_model=None, file_processor:MultiFormatProcessor=None):
        """
        Initialize semantic chunker with shared models.
        
        Args:
            model_name: Name of sentence transformer model for semantic similarity (deprecated, use embedding_model)
            language: Language setting
            embedding_model: Shared embedding model instance
            text_generation_model: Shared text generation model instance
        """
        self.logger = logger.bind(name="SemanticChunker")
        self.model_name = get_config("rag.embedding.minilm.model_name", "sentence-transformers/all-MiniLM-L6-v2")
        # Use shared embedding model if provided
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        # if embedding_model:
        #     self.model = embedding_model.model
        #     self.model_name = embedding_model.model_name
        #     self.logger.info("✅ Using shared embedding model for semantic chunking")
        # else:
        #     self.logger.error("No shared embedding model provided, using default model")
        #     return False
        
        # Setting language
        self.language = language
        self.file_processor = file_processor
        
        # Chunking parameters
        self.min_chunk_size = get_config("rag.semantic_chunking.min_chunk_size", 100)
        self.max_chunk_size = get_config("rag.semantic_chunking.max_chunk_size", 1000)
        self.similarity_threshold = get_config("rag.semantic_chunking.semantic_similarity_threshold", 0.7)
        self.overlap_size = get_config("rag.chunk_overlap", 50)
        
        # Dynamic windowing parameters
        self.dynamic_window_enabled = get_config("rag.semantic_chunking.dynamic_window_enabled", True)
        self.base_chunk_size = get_config("rag.semantic_chunking.base_chunk_size", 512)
        self.complexity_multiplier = get_config("rag.semantic_chunking.complexity_multiplier", 1.5)
        self.similarity_adaptive = get_config("rag.semantic_chunking.similarity_adaptive", True)
        self.content_type_weights = {
            'code': 1.2,           # Code blocks need more space
            'api_reference': 1.1,  # API docs are dense
            'header': 0.8,         # Headers are short
            'text': 1.0,           # Regular text
            'example': 1.3         # Examples need more context
        }
        
        # Boost.Asio specific patterns
        self.code_patterns = {
            'cpp_function': r'^\s*(?:template\s*<[^>]*>\s*)?(?:inline\s+)?(?:static\s+)?(?:const\s+)?(?:volatile\s+)?(?:explicit\s+)?(?:virtual\s+)?(?:friend\s+)?(?:constexpr\s+)?(?:noexcept\s*\([^)]*\)\s*)?(?:override\s+)?(?:final\s+)?(?:class\s+|struct\s+|enum\s+|union\s+)?\w+(?:\s*<[^>]*>)?\s+(?:\*|\&)?\s*\w+\s*\([^)]*\)\s*(?:const\s*)?(?:\s*=\s*(?:0|default|delete))?\s*[;{]',
            'cpp_include': r'^\s*#\s*include\s*[<"][^>"]*[>"]',
            'cpp_define': r'^\s*#\s*define\s+\w+',
            'cpp_namespace': r'^\s*namespace\s+\w+',
            'cpp_class': r'^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+\w+',
            'cpp_comment': r'^\s*//.*$',
            'cpp_block_comment': r'/\*.*?\*/',
        }
        
        self.markdown_patterns = {
            'header': r'^#{1,6}\s+.+',
            'code_block': r'```[\s\S]*?```',
            'inline_code': r'`[^`]+`',
            'list_item': r'^\s*[-*+]\s+',
            'numbered_list': r'^\s*\d+\.\s+',
            'table_row': r'^\s*\|.*\|',
            'link': r'\[([^\]]+)\]\(([^)]+)\)',
        }
        
        if embedding_model:
            self.logger.info(f"✅ SemanticChunker initialized with shared embedding model: {self.model_name}")
        else:
            self.logger.info(f"SemanticChunker initialized with model: {self.model_name}")
    
    
    def detect_content_type(self, text: str) -> str:
        """
        Detect the type of content in a text block.
        
        Args:
            text: Text to analyze
            
        Returns:
            Content type: 'code', 'header', 'text', 'example', 'api_reference'
        """
        text = text.strip()
        
        # Check for code patterns
        for pattern_name, pattern in self.code_patterns.items():
            if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                if 'function' in pattern_name or 'class' in pattern_name:
                    return 'api_reference'
                return 'code'
        
        # Check for markdown patterns
        if re.match(self.markdown_patterns['header'], text):
            return 'header'
        
        if re.search(self.markdown_patterns['code_block'], text):
            return 'code'
        
        if re.search(self.markdown_patterns['inline_code'], text):
            return 'api_reference'
        
        # Check for example indicators
        example_indicators = ['example', 'demo', 'sample', 'usage', 'tutorial']
        if any(indicator in text.lower() for indicator in example_indicators):
            return 'example'
        
        return 'text'
    
    def extract_function_name(self, text: str) -> Optional[str]:
        """Extract function name from C++ code."""
        # Simple function name extraction
        func_match = re.search(r'(\w+)\s*\([^)]*\)', text)
        if func_match:
            return func_match.group(1)
        return None
    
    def split_by_structure(self, text: str, source_file: str) -> List[Tuple[str, ChunkMetadata]]:
        """
        Split text by structural elements (headers, code blocks, etc.).
        
        Args:
            text: Text to split
            source_file: Source file path
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_metadata = None
        line_number = 0
        
        for i, line in enumerate(lines):
            line_number = i + 1
            content_type = self.detect_content_type(line)
            
            # Start new chunk if content type changes significantly
            if (current_metadata and 
                current_metadata.chunk_type != content_type and 
                current_chunk and
                len('\n'.join(current_chunk)) > self.min_chunk_size):
                
                # Finalize current chunk
                chunk_text = '\n'.join(current_chunk)
                current_metadata.end_line = line_number - 1
                chunks.append((chunk_text, current_metadata))
                
                # Start new chunk
                current_chunk = [line]
                current_metadata = ChunkMetadata(
                    chunk_id=f"{source_file}_{line_number}",
                    source_file=source_file,
                    chunk_type=content_type,
                    start_line=line_number,
                    end_line=line_number,
                    function_name=self.extract_function_name(line) if content_type == 'api_reference' else None
                )
            else:
                # Add line to current chunk
                if not current_metadata:
                    current_metadata = ChunkMetadata(
                        chunk_id=f"{source_file}_{line_number}",
                        source_file=source_file,
                        chunk_type=content_type,
                        start_line=line_number,
                        end_line=line_number,
                        function_name=self.extract_function_name(line) if content_type == 'api_reference' else None
                    )
                
                current_chunk.append(line)
                current_metadata.end_line = line_number
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            current_metadata.end_line = line_number
            chunks.append((chunk_text, current_metadata))
        
        return chunks
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text chunks.
        
        Args:
            text1: First text chunk
            text2: Second text chunk
            
        Returns:
            Similarity score between 0 and 1
        """
        
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def calculate_content_complexity(self, text: str) -> float:
        """
        Calculate content complexity score for dynamic windowing.
        
        Args:
            text: Text to analyze
            
        Returns:
            Complexity score (0.5 to 2.0)
        """
        complexity_score = 1.0
        
        # Count code elements
        code_elements = len(re.findall(r'```[\s\S]*?```', text))
        inline_code = len(re.findall(r'`[^`]+`', text))
        functions = len(re.findall(r'\w+\s*\([^)]*\)', text))
        classes = len(re.findall(r'(?:class|struct)\s+\w+', text))
        
        # Count structural elements
        headers = len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE))
        lists = len(re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE))
        tables = len(re.findall(r'^\s*\|.*\|', text, re.MULTILINE))
        
        # Calculate complexity based on content density
        total_elements = code_elements + inline_code + functions + classes + headers + lists + tables
        text_length = len(text)
        
        if text_length > 0:
            element_density = total_elements / (text_length / 100)  # Elements per 100 chars
            
            if element_density > 2.0:
                complexity_score = 1.8  # Very complex
            elif element_density > 1.5:
                complexity_score = 1.5  # Complex
            elif element_density > 1.0:
                complexity_score = 1.2  # Moderately complex
            elif element_density < 0.3:
                complexity_score = 0.7  # Simple
        
        return max(0.5, min(2.0, complexity_score))
    
    def calculate_dynamic_chunk_size(self, content_type: str, complexity: float, 
                                   previous_similarity: float = 0.5) -> int:
        """
        Calculate dynamic chunk size based on content type, complexity, and context.
        
        Args:
            content_type: Type of content ('code', 'header', 'text', etc.)
            complexity: Content complexity score
            previous_similarity: Similarity with previous chunk
            
        Returns:
            Dynamic chunk size
        """
        if not self.dynamic_window_enabled:
            return self.base_chunk_size
        
        # Base size from content type
        type_weight = self.content_type_weights.get(content_type, 1.0)
        base_size = int(self.base_chunk_size * type_weight)
        
        # Adjust for complexity
        complexity_adjusted = int(base_size * complexity)
        
        # Adjust for semantic similarity (if adaptive similarity is enabled)
        if self.similarity_adaptive:
            if previous_similarity > 0.8:
                # High similarity - can use smaller chunks
                similarity_factor = 0.8
            elif previous_similarity < 0.3:
                # Low similarity - need larger chunks for context
                similarity_factor = 1.3
            else:
                similarity_factor = 1.0
            
            complexity_adjusted = int(complexity_adjusted * similarity_factor)
        
        # Ensure within bounds
        return max(self.min_chunk_size, min(self.max_chunk_size, complexity_adjusted))
    
    def calculate_dynamic_similarity_threshold(self, content_type: str, 
                                             chunk_size: int) -> float:
        """
        Calculate dynamic similarity threshold based on content type and chunk size.
        
        Args:
            content_type: Type of content
            chunk_size: Size of current chunk
            
        Returns:
            Dynamic similarity threshold
        """
        if not self.similarity_adaptive:
            return self.similarity_threshold
        
        base_threshold = self.similarity_threshold
        
        # Adjust threshold based on content type
        if content_type == 'code':
            # Code blocks should be more strictly similar
            return min(0.9, base_threshold + 0.1)
        elif content_type == 'header':
            # Headers can be less similar
            return max(0.5, base_threshold - 0.1)
        elif content_type == 'api_reference':
            # API references need moderate similarity
            return base_threshold
        else:
            return base_threshold
    
    def merge_similar_chunks(self, chunks: List[Tuple[str, ChunkMetadata]]) -> List[Tuple[str, ChunkMetadata]]:
        """
        Merge semantically similar chunks using dynamic windowing.
        
        Args:
            chunks: List of (chunk_text, metadata) tuples
            
        Returns:
            Merged chunks with dynamic sizing
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk_text = chunks[0][0]
        current_metadata = chunks[0][1]
        previous_similarity = 0.5  # Default similarity for first chunk
        
        for i in range(1, len(chunks)):
            next_chunk_text = chunks[i][0]
            next_metadata = chunks[i][1]
            
            # Calculate dynamic parameters
            current_complexity = self.calculate_content_complexity(current_chunk_text)
            dynamic_chunk_size = self.calculate_dynamic_chunk_size(
                current_metadata.chunk_type, 
                current_complexity, 
                previous_similarity
            )
            dynamic_threshold = self.calculate_dynamic_similarity_threshold(
                current_metadata.chunk_type, 
                len(current_chunk_text)
            )
            
            # Calculate similarity with next chunk
            similarity = self.calculate_semantic_similarity(current_chunk_text, next_chunk_text)
            
            # Check if chunks should be merged using dynamic criteria
            should_merge = False
            
            # Merge if current chunk is smaller than dynamic size
            if len(current_chunk_text) < dynamic_chunk_size:
                should_merge = True
            
            # Merge if chunks are semantically similar above dynamic threshold
            elif (current_metadata.chunk_type == next_metadata.chunk_type and
                  similarity > dynamic_threshold):
                should_merge = True
            
            # Merge if both chunks are small relative to their dynamic sizes
            elif (len(current_chunk_text) < dynamic_chunk_size * 1.2 and
                  len(next_chunk_text) < dynamic_chunk_size * 1.2 and
                  current_metadata.chunk_type == next_metadata.chunk_type):
                should_merge = True
            
            if should_merge:
                # Merge chunks
                current_chunk_text += '\n' + next_chunk_text
                current_metadata.end_line = next_metadata.end_line
                
                # Update chunk ID to reflect merged nature
                current_metadata.chunk_id = f"{current_metadata.chunk_id}_{next_metadata.chunk_id}"
                
                # Update similarity for next iteration
                previous_similarity = similarity
            else:
                # Finalize current chunk and start new one
                merged_chunks.append((current_chunk_text, current_metadata))
                current_chunk_text = next_chunk_text
                current_metadata = next_metadata
                previous_similarity = 0.5  # Reset for new chunk
        
        # Add final chunk
        merged_chunks.append((current_chunk_text, current_metadata))
        
        return merged_chunks
    
    def split_large_chunks(self, chunks: List[Tuple[str, ChunkMetadata]]) -> List[Tuple[str, ChunkMetadata]]:
        """
        Split chunks that are too large using dynamic windowing.
        
        Args:
            chunks: List of (chunk_text, metadata) tuples
            
        Returns:
            Split chunks with dynamic sizing
        """
        split_chunks = []
        
        for chunk_text, metadata in chunks:
            # Calculate dynamic max size for this content type
            complexity = self.calculate_content_complexity(chunk_text)
            dynamic_max_size = self.calculate_dynamic_chunk_size(
                metadata.chunk_type, 
                complexity, 
                0.5  # Default similarity
            )
            
            if len(chunk_text) <= dynamic_max_size:
                split_chunks.append((chunk_text, metadata))
                continue
            
            # Split large chunks by sentences or logical breaks
            lines = chunk_text.split('\n')
            current_chunk_lines = []
            current_length = 0
            chunk_index = 0
            
            for line in lines:
                line_length = len(line) + 1  # +1 for newline
                
                if current_length + line_length > dynamic_max_size and current_chunk_lines:
                    # Create new chunk
                    new_chunk_text = '\n'.join(current_chunk_lines)
                    new_metadata = ChunkMetadata(
                        chunk_id=f"{metadata.chunk_id}_part_{chunk_index}",
                        source_file=metadata.source_file,
                        chunk_type=metadata.chunk_type,
                        start_line=metadata.start_line + chunk_index * 100,  # Approximate
                        end_line=metadata.start_line + chunk_index * 100 + len(current_chunk_lines),
                        parent_section=metadata.parent_section,
                        function_name=metadata.function_name,
                        example_type=metadata.example_type,
                        importance_score=metadata.importance_score
                    )
                    split_chunks.append((new_chunk_text, new_metadata))
                    
                    # Start new chunk with dynamic overlap
                    overlap_size = min(self.overlap_size, dynamic_max_size // 10)  # 10% overlap
                    overlap_lines = current_chunk_lines[-overlap_size//50:] if overlap_size > 0 else []
                    current_chunk_lines = overlap_lines + [line]
                    current_length = sum(len(l) + 1 for l in current_chunk_lines)
                    chunk_index += 1
                else:
                    current_chunk_lines.append(line)
                    current_length += line_length
            
            # Add final chunk
            if current_chunk_lines:
                new_chunk_text = '\n'.join(current_chunk_lines)
                new_metadata = ChunkMetadata(
                    chunk_id=f"{metadata.chunk_id}_part_{chunk_index}",
                    source_file=metadata.source_file,
                    chunk_type=metadata.chunk_type,
                    start_line=metadata.start_line + chunk_index * 100,
                    end_line=metadata.end_line,
                    parent_section=metadata.parent_section,
                    function_name=metadata.function_name,
                    example_type=metadata.example_type,
                    importance_score=metadata.importance_score
                )
                split_chunks.append((new_chunk_text, new_metadata))
        
        return split_chunks
    
    def chunk_document(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Chunk a document using semantic and structural awareness.
        
        Args:
            text: Document text to chunk
            source_file: Source file path
            
        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            self.logger.info(f"Chunking document: {source_file}")
            
            # Step 1: Split by structure
            structural_chunks = self.split_by_structure(text, source_file)
            self.logger.info(f"Created {len(structural_chunks)} structural chunks")
            
            # Step 2: Merge similar small chunks
            merged_chunks = self.merge_similar_chunks(structural_chunks)
            self.logger.info(f"Merged to {len(merged_chunks)} chunks")
            
            # Step 3: Split large chunks
            final_chunks = self.split_large_chunks(merged_chunks)
            self.logger.info(f"Final chunk count: {len(final_chunks)}")
            
            # Convert to dictionary format
            chunk_dicts = []
            for i, (chunk_text, metadata) in enumerate(final_chunks):
                chunk_dict = {
                    'chunk_id': metadata.chunk_id,
                    'text': chunk_text,
                    'metadata': {
                        'source_file': metadata.source_file,
                        'chunk_type': metadata.chunk_type,
                        'start_line': metadata.start_line,
                        'end_line': metadata.end_line,
                        'parent_section': metadata.parent_section,
                        'function_name': metadata.function_name,
                        'example_type': metadata.example_type,
                        'importance_score': metadata.importance_score,
                        'chunk_index': i,
                        'chunk_length': len(chunk_text)
                    }
                }
                chunk_dicts.append(chunk_dict)
            
            return chunk_dicts
            
        except Exception as e:
            self.logger.error(f"Error chunking document {source_file}: {e}")
            return []
    
    def process_knowledge_base(self, raw_file_list: List[str] = None, max_files: int = None) -> Tuple[bool, List[str]]:
        """
        Process entire knowledge base with semantic chunking.
        
        Args:
            raw_file_list: List of raw file paths
            
        Returns:
            True if successful, False otherwise
        """
        try:
            input_dir = f"{get_config('data.source_data.new_data_path', 'data/source_data/new')}/{self.language}"
            self.input_path = Path(input_dir)
            if raw_file_list is None:
                raw_file_list = self.file_processor.get_file_list(self.input_path)
            if max_files is not None:
                raw_file_list = raw_file_list[:max_files]
            
            output_dir = f"{get_config('data.processed_data.chunked_data_path', 'data/processed/chunked')}/{self.language}"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process all files
            total_chunks = 0
            processed_files = 0
            chunk_file_list = []
            
            for file_path in raw_file_list:
                text = self.file_processor.extract_text_from_file(file_path)
                if text != "":
                    relative_path = file_path.relative_to(self.input_path)
                    self.logger.info(f"Processing: {relative_path}")
                    output_file = output_path / f"{file_path.stem}_semantic_chunks.json"
                    if output_file.exists():
                        self.logger.info(f"Skipping {relative_path} because it already exists")
                        continue
                    chunks = self.chunk_document(text, str(relative_path))
                    if chunks:
                        # Save chunks
                        output_file = output_path / f"{file_path.stem}_semantic_chunks.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(chunks, f, ensure_ascii=False, indent=2)
                        chunk_file_list.append(output_file)
                        total_chunks += len(chunks)
                        processed_files += 1
            
            self.logger.info(f"Processed {processed_files} files, created {total_chunks} semantic chunks")
            return True, chunk_file_list
            
        except Exception as e:
            self.logger.error(f"Error processing knowledge base: {e}")
            return False, chunk_file_list


def main():
    """Main function for command-line usage."""
    input_dir = get_config("rag.clead_database_dir", "data/processed/knowledge_base")
    output_dir = get_config("rag.semantic_chunks_dir", "data/processed/semantic_chunks")
    model_name = get_config("rag.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    
    chunker = SemanticChunker(model_name=model_name)
    
    if chunker.process_knowledge_base(input_dir, output_dir):
        print("Semantic chunking completed successfully!")
    else:
        print("Semantic chunking failed!")
        exit(1)


if __name__ == "__main__":
    main()
