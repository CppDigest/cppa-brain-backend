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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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


class HeadingAwareChunker:
    """
    Chunk documents while preserving structure.
    - Respect heading boundaries
    - Preserve code blocks
    - Include section titles in chunks
    - Add metadata fields (library, module, heading_path)
    - Handle code files differently (split by classes, avoid splitting short files)
    """

    def __init__(
        self, min_chunk_size: int = 100, max_chunk_size: int = 1000, overlap: int = 50
    ):
        self.logger = logger.bind(name="HeadingAwareChunker")
        self.min_chunk_size = min_chunk_size or get_config(
            "rag.chunking.min_chunk_size", 100
        )
        self.max_chunk_size = max_chunk_size or get_config(
            "rag.chunking.max_chunk_size", 1000
        )
        self.overlap = overlap

        # Code file extensions
        self.code_extensions = {
            ".cpp",
            ".c",
            ".hpp",
            ".h",
            ".cc",
            ".cxx",
            ".c++",
            ".py",
            ".java",
            ".js",
            ".ts",
            ".cs",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".swift",
            ".kt",
            ".scala",
            ".clj",
            ".hs",
            ".ml",
            ".fs",
            ".vb",
            ".dart",
            ".r",
            ".m",
            ".mm",
            ".cu",
            ".cl",
            ".f",
            ".f90",
            ".f95",
            ".f03",
            ".f08",
            ".f15",
            ".f18",
            ".f23",
        }

        # End markers that should not start a chunk
        self.end_markers = {
            "#endif",
            "#else",
            "#elif",
            "}",
            "]",
            ")",
            "@endcode",
            "@end",
            "@endverbatim",
            "@endparblock",
            "end if",
            "end for",
            "end while",
            "end function",
            "end class",
            "end struct",
            "end namespace",
            "end try",
            "end catch",
            "end finally",
            "end switch",
            "end case",
            "end default",
        }

    def chunk_document(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk document with heading awareness.

        Args:
            text: Document text
            metadata: Document metadata

        Returns:
            List of chunk dicts with text and metadata
        """
        if metadata is None:
            metadata = {}

        # Parse document structure based on file type
        source_file = metadata.get("source_url", "")
        sections = self._parse_sections(text, source_file)

        # Create chunks from sections
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, metadata)
            chunks.extend(section_chunks)

        self.logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks

    def _parse_sections(self, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """
        Parse document into sections based on file type.
        - For code files: parse by classes/functions
        - For documentation: parse by headings (markdown, RST)
        """
        if self._is_code_file(source_file):
            return self._parse_code_sections(text, source_file)
        else:
            return self._parse_documentation_sections(text)

    def _parse_code_sections(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Parse code files into logical sections (classes, functions, etc.).
        """
        sections = []
        lines = text.split("\n")

        # Remove source URL if present
        if "Source URL: " in lines[0]:
            lines = lines[1:]

        # For code files, create a single section with the entire content
        # The actual splitting will be handled in _chunk_code_file
        section = {
            "heading": f"Code File: {Path(source_file).name}",
            "level": 1,
            "content": lines,
            "heading_path": [Path(source_file).stem],
            "file_type": "code",
        }
        sections.append(section)

        return sections

    def _parse_documentation_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse documentation files into sections based on headings.
        Supports markdown and restructuredText headings.
        """
        sections = []

        # Patterns for headings
        heading_patterns = [
            (r"^(#{1,6})\s+(.+)$", "markdown"),  # Markdown: # Heading
            (r"^(.+)\n([=\-~^]{3,})$", "rst"),  # RST: Heading\n====
            (r"^(\d+\.)+\s+(.+)$", "numbered"),  # Numbered: 1.2.3 Heading
        ]

        lines = text.split("\n")
        if "Source URL: " in lines[0]:
            lines = lines[1:]
        current_section = {"heading": "", "level": 0, "content": [], "heading_path": []}
        heading_stack = []

        i = 0
        while i < len(lines):
            line = lines[i]
            heading_found = False

            # Check for headings
            for pattern, heading_type in heading_patterns:
                if heading_type == "rst" and i + 1 < len(lines):
                    # RST needs to check next line
                    match = re.match(r"^(.+)$", line)
                    next_match = re.match(r"^([=\-~^]{3,})$", lines[i + 1])

                    if match and next_match:
                        heading_text = match.group(1).strip()
                        underline_char = next_match.group(1)[0]

                        # Determine level based on underline character
                        level_map = {"=": 1, "-": 2, "~": 3, "^": 4}
                        level = level_map.get(underline_char, 2)

                        # Start new section
                        if current_section["content"]:
                            sections.append(current_section)

                        # Update heading stack
                        while heading_stack and heading_stack[-1]["level"] >= level:
                            heading_stack.pop()

                        heading_path = [h["heading"] for h in heading_stack] + [
                            heading_text
                        ]

                        current_section = {
                            "heading": heading_text,
                            "level": level,
                            "content": [],
                            "heading_path": heading_path,
                        }

                        heading_stack.append({"heading": heading_text, "level": level})

                        i += 2  # Skip underline
                        heading_found = True
                        break

                else:
                    match = re.match(pattern, line)
                    if match:
                        if heading_type == "markdown":
                            level = len(match.group(1))
                            heading_text = match.group(2).strip()
                        elif heading_type == "numbered":
                            level = match.group(1).count(".")
                            heading_text = match.group(2).strip()
                        else:
                            continue

                        # Start new section
                        if current_section["content"]:
                            sections.append(current_section)

                        # Update heading stack
                        while heading_stack and heading_stack[-1]["level"] >= level:
                            heading_stack.pop()

                        heading_path = [h["heading"] for h in heading_stack] + [
                            heading_text
                        ]

                        current_section = {
                            "heading": heading_text,
                            "level": level,
                            "content": [],
                            "heading_path": heading_path,
                        }

                        heading_stack.append({"heading": heading_text, "level": level})

                        i += 1
                        heading_found = True
                        break

            if not heading_found:
                current_section["content"].append(line)
                i += 1

        # Add last section
        if current_section["content"]:
            sections.append(current_section)

        return sections

    def _is_code_file(self, source_file: str) -> bool:
        """Check if the file is a code file based on extension."""
        if not source_file:
            return False
        file_path = Path(source_file)
        return file_path.suffix.lower() in self.code_extensions

    def _count_lines(self, text: str) -> int:
        """Count the number of lines in text."""
        return len(text.split("\n"))

    def _starts_with_end_marker(self, line: str) -> bool:
        """Check if a line starts with an end marker."""
        line = line.strip()
        for marker in self.end_markers:
            if line.startswith(marker):
                return True
        return False

    def _find_safe_split_point(
        self, lines: List[str], start_idx: int, max_size: int
    ) -> int:
        """
        Find a safe split point that doesn't start with end markers.
        Returns the index where to split.
        """
        if start_idx >= len(lines):
            return len(lines)

        current_size = 0
        best_split = start_idx

        for i in range(start_idx, len(lines)):
            line_size = len(lines[i]) + 1  # +1 for newline
            current_size += line_size

            # If we've exceeded max size, look for a good split point
            if current_size > max_size:
                # Look backwards for a safe split point
                for j in range(i, start_idx, -1):
                    if not self._starts_with_end_marker(lines[j]):
                        # Found a line that doesn't start with end marker
                        # Check if it's a good split point (ends with ;, }, or is empty)
                        if (
                            lines[j].strip().endswith(";")
                            or lines[j].strip().endswith("}")
                            or lines[j].strip() == ""
                            or lines[j].strip().endswith(":")
                        ):
                            return j + 1

                # If no good split point found, split at current position anyway
                return i

        return len(lines)

    def _split_by_classes(self, text: str) -> List[Dict[str, Any]]:
        """
        Split code text by class definitions.
        Returns list of class chunks with metadata.
        """
        lines = text.split("\n")
        class_chunks = []
        current_class = []
        current_class_name = None
        brace_count = 0
        in_class = False

        for i, line in enumerate(lines):
            # Check for class/struct definition
            class_match = re.match(
                r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+(\w+)", line
            )
            if class_match and not in_class:
                # Save previous class if exists
                if current_class and current_class_name:
                    class_chunks.append(
                        {
                            "content": "\n".join(current_class),
                            "class_name": current_class_name,
                            "start_line": i - len(current_class) + 1,
                            "end_line": i,
                        }
                    )

                # Start new class
                current_class = [line]
                current_class_name = class_match.group(1)
                in_class = True
                brace_count = 0
            elif in_class:
                current_class.append(line)

                # Count braces to track class boundaries
                brace_count += line.count("{") - line.count("}")

                # Class ends when brace count reaches 0
                if brace_count == 0 and "{" in line:
                    class_chunks.append(
                        {
                            "content": "\n".join(current_class),
                            "class_name": current_class_name,
                            "start_line": i - len(current_class) + 1,
                            "end_line": i + 1,
                        }
                    )
                    current_class = []
                    current_class_name = None
                    in_class = False

        # Add final class if exists
        if current_class and current_class_name:
            class_chunks.append(
                {
                    "content": "\n".join(current_class),
                    "class_name": current_class_name,
                    "start_line": len(lines) - len(current_class) + 1,
                    "end_line": len(lines),
                }
            )

        return class_chunks

    def _chunk_section(
        self, section: Dict[str, Any], doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk a section while preserving code blocks.
        Handle code files differently: split by classes and avoid splitting short files.
        """
        content = "\n".join(section["content"])
        source_file = doc_metadata.get("source_url", "")

        # Check if this is a code file
        if self._is_code_file(source_file):
            return self._chunk_code_file(content, section, doc_metadata)

        # Original logic for non-code files
        # Extract and preserve code blocks
        code_blocks = []

        def extract_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        # Match code blocks (```...``` or indented)
        content_no_code = re.sub(r"```[\s\S]*?```", extract_code, content)
        content_no_code = re.sub(
            r"(?:^|\n)(    .+(?:\n    .+)*)", extract_code, content_no_code
        )

        # Split into chunks
        chunks = []
        words = content_no_code.split()

        if len(words) <= self.max_chunk_size:
            # Single chunk
            chunk_text = self._restore_code_blocks(content_no_code, code_blocks)
            chunks.append(self._create_chunk(section, chunk_text, doc_metadata, 0))
        else:
            # Multiple chunks with overlap
            start = 0
            chunk_idx = 0

            while start < len(words):
                end = min(start + self.max_chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                chunk_text = self._restore_code_blocks(chunk_text, code_blocks)

                chunks.append(
                    self._create_chunk(section, chunk_text, doc_metadata, chunk_idx)
                )

                chunk_idx += 1
                start = end - self.overlap if end < len(words) else end

        return chunks

    def _chunk_code_file(
        self, content: str, section: Dict[str, Any], doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Handle chunking for code files.
        - Split by class definitions if multiple classes exist
        - Avoid splitting short files (below 50 lines) unless necessary
        """
        line_count = self._count_lines(content)

        # For short code files (below 50 lines), avoid splitting unless absolutely necessary
        if line_count < 50:
            # Only split if the content is extremely long (over 10x max_chunk_size)
            if len(content) > self.max_chunk_size * 10:
                # Split by sentences or logical breaks, but preserve structure
                return self._split_long_short_code(content, section, doc_metadata)
            else:
                # Keep as single chunk
                return [self._create_chunk(section, content, doc_metadata, 0)]

        # For longer code files, try to split by classes first
        class_chunks = self._split_by_classes(content)

        if len(class_chunks) > 1:
            # Multiple classes found - create chunks for each class
            chunks = []
            for i, class_chunk in enumerate(class_chunks):
                # Create a modified section for this class
                class_section = section.copy()
                class_section["content"] = class_chunk["content"].split("\n")
                class_section["class_name"] = class_chunk["class_name"]

                # Add class-specific metadata
                class_metadata = doc_metadata.copy()
                class_metadata["class_name"] = class_chunk["class_name"]
                class_metadata["start_line"] = class_chunk["start_line"]
                class_metadata["end_line"] = class_chunk["end_line"]

                chunks.append(
                    self._create_chunk(
                        class_section, class_chunk["content"], class_metadata, i
                    )
                )

            return chunks
        else:
            # Single class or no classes - use regular chunking but with code-aware logic
            return self._chunk_single_class_code(content, section, doc_metadata)

    def _split_long_short_code(
        self, content: str, section: Dict[str, Any], doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split long short code files by logical breaks."""
        lines = content.split("\n")
        chunks = []
        chunk_idx = 0
        start_idx = 0

        while start_idx < len(lines):
            # Find safe split point
            end_idx = self._find_safe_split_point(lines, start_idx, self.max_chunk_size)

            # Extract chunk
            chunk_lines = lines[start_idx:end_idx]
            chunk_text = "\n".join(chunk_lines)

            # Ensure chunk doesn't start with end marker
            if chunk_lines and self._starts_with_end_marker(chunk_lines[0]):
                # If chunk starts with end marker, try to include previous line
                if start_idx > 0:
                    chunk_lines = lines[start_idx - 1 : end_idx]
                    chunk_text = "\n".join(chunk_lines)
                    start_idx -= 1

            chunks.append(
                self._create_chunk(section, chunk_text, doc_metadata, chunk_idx)
            )
            chunk_idx += 1
            start_idx = end_idx

        return chunks

    def _chunk_single_class_code(
        self, content: str, section: Dict[str, Any], doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle chunking for single class or non-class code files."""
        # Use regular chunking logic but with code-aware splitting
        words = content.split()

        if len(words) <= self.max_chunk_size:
            return [self._create_chunk(section, content, doc_metadata, 0)]

        # Split by logical code boundaries using safe split points
        lines = content.split("\n")
        chunks = []
        chunk_idx = 0
        start_idx = 0

        while start_idx < len(lines):
            # Find safe split point
            end_idx = self._find_safe_split_point(lines, start_idx, self.max_chunk_size)

            # Extract chunk
            chunk_lines = lines[start_idx:end_idx]
            chunk_text = "\n".join(chunk_lines)

            # Ensure chunk doesn't start with end marker
            if chunk_lines and self._starts_with_end_marker(chunk_lines[0]):
                # If chunk starts with end marker, try to include previous line
                if start_idx > 0:
                    chunk_lines = lines[start_idx - 1 : end_idx]
                    chunk_text = "\n".join(chunk_lines)
                    start_idx -= 1

            chunks.append(
                self._create_chunk(section, chunk_text, doc_metadata, chunk_idx)
            )
            chunk_idx += 1
            start_idx = end_idx

        return chunks

    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """Restore code block placeholders with actual code."""
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", code_block)
        return text

    def _detect_chunk_type(self, chunk_text: str) -> str:
        """
        Detect chunk type: 'code', 'text'.

        Args:
            chunk_text: Text content of the chunk
            section: Section metadata

        Returns:
            Chunk type string
        """
        # Check for code blocks
        if re.search(r"```[\s\S]*?```", chunk_text):
            return "code"

        # Check for inline code density (if lots of inline code, might be code-heavy)
        inline_code_count = len(re.findall(r"`[^`]+`", chunk_text))
        if inline_code_count > 5:
            return "code"

        # --- Heuristic detection for code-like syntax ---
        code_indicators = [
            ";",
            "{",
            "}",
            "=>",
            "::",
            "->",
            "#include",
            "using namespace",
            "std::",
            "def ",
            "class ",
            "import ",
            "return ",
            "if ",
            "for ",
            "while ",
            "int ",
            "float ",
            "void ",
        ]
        lines = chunk_text.strip().splitlines()

        # Heuristic: how many lines look like code
        code_like_lines = sum(
            any(tok in line for tok in code_indicators) for line in lines
        )
        ratio = code_like_lines / max(len(lines), 1)

        # If more than half of the lines look like code, or the text is dense with code tokens
        if ratio > 0.5 or (ratio > 0.2 and len(lines) > 2):
            return "code"

        # Also, short single-line code (e.g. `std::cout << i;`)
        if len(lines) == 1 and any(tok in chunk_text for tok in code_indicators):
            return "code"

        # Default to text
        return "text"

    def _create_chunk(
        self,
        section: Dict[str, Any],
        chunk_text: str,
        doc_metadata: Dict[str, Any],
        chunk_idx: int,
    ) -> Dict[str, Any]:
        """Create chunk with metadata."""
        # Detect chunk type
        chunk_type = self._detect_chunk_type(chunk_text)

        # Generate chunk_id
        source_url = doc_metadata.get("source_url", "")
        heading_path = (
            " > ".join(section["heading_path"]) if section["heading_path"] else "root"
        )
        chunk_id_str = f"{source_url}_{heading_path}_{chunk_idx}"
        chunk_id = hashlib.md5(chunk_id_str.encode(), usedforsecurity=False).hexdigest()

        # Extract metadata
        chunk_metadata = doc_metadata.copy()
        chunk_metadata.update(
            {
                "section": section["heading"],
                "heading_path": " > ".join(section["heading_path"]),
                "heading_level": section["level"],
                "chunk_index": chunk_idx,
                "has_code": "```" in chunk_text or "    " in chunk_text,
                "chunk_type": chunk_type,
            }
        )

        # Add class-specific metadata if available
        if "class_name" in section:
            chunk_metadata["class_name"] = section["class_name"]
        if "class_name" in doc_metadata:
            chunk_metadata["class_name"] = doc_metadata["class_name"]
        if "start_line" in doc_metadata:
            chunk_metadata["start_line"] = doc_metadata["start_line"]
        if "end_line" in doc_metadata:
            chunk_metadata["end_line"] = doc_metadata["end_line"]

        # Try to extract library/module info
        chunk_metadata["library"] = self._extract_library(source_url)
        chunk_metadata["module"] = self._extract_module(
            source_url, section["heading_path"]
        )

        return {
            "chunk_id": chunk_id,
            "text": chunk_text.strip(),
            "metadata": chunk_metadata,
        }

    def _extract_library(self, source_file: str) -> str:
        """Extract library name from source file path."""
        # Look for boost library name in path
        match = re.search(r"boost[/\\]([a-z_]+)", source_file.lower())
        if match:
            return match.group(1)
        return ""

    def _extract_module(self, source_file: str, heading_path: List[str]) -> str:
        """Extract module name from path or headings."""
        # Try to get from filename
        filename = Path(source_file).stem
        if filename:
            return filename

        # Try to get from first heading
        if heading_path:
            return heading_path[0]

        return ""


class BoilerplateRemover:
    """Remove navigation, headers, footers, and other boilerplate."""

    def __init__(self):
        self.logger = logger.bind(name="BoilerplateRemover")

        # Common boilerplate patterns
        self.boilerplate_patterns = [
            r"(?i)copyright\s+©.*",
            r"(?i)all\s+rights\s+reserved.*",
            r"(?i)terms\s+(of\s+service|and\s+conditions).*",
            r"(?i)privacy\s+policy.*",
            r"(?i)cookie\s+policy.*",
            r"(?i)subscribe\s+to.*newsletter.*",
            r"(?i)follow\s+us\s+on.*",
            r"(?i)(home|about|contact|sitemap)\s*\|.*",  # Navigation
            r"(?i)last\s+updated:.*",
            r"(?i)page\s+\d+\s+of\s+\d+.*",
        ]

    def remove_boilerplate(self, text: str) -> str:
        """Remove boilerplate content from text."""
        cleaned = text

        for pattern in self.boilerplate_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        # Remove excessive whitespace
        cleaned = re.sub(r"\n\n\n+", "\n\n", cleaned)
        cleaned = re.sub(r" +", " ", cleaned)

        return cleaned.strip()

    def clean_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean boilerplate from all chunks."""
        cleaned_chunks = []

        for chunk in chunks:
            chunk_copy = chunk.copy()
            chunk_copy["text"] = self.remove_boilerplate(chunk["text"])

            # Only keep chunks with meaningful content
            if len(chunk_copy["text"].split()) >= 10:  # At least 10 words
                cleaned_chunks.append(chunk_copy)

        removed = len(chunks) - len(cleaned_chunks)
        if removed > 0:
            self.logger.debug(f"🗑️ Removed {removed} boilerplate chunks")

        return cleaned_chunks


class DynamicOverlapChunker:
    """Adjust overlap based on section density and complexity."""

    def __init__(self):
        self.logger = logger.bind(name="DynamicOverlapChunker")

    def calculate_overlap(self, section_text: str, base_overlap: int = 50) -> int:
        """
        Calculate optimal overlap for a section.
        Dense/complex sections get more overlap to preserve context.
        """
        # Calculate section density (technical terms per 100 words)
        words = section_text.split()
        if len(words) == 0:
            return base_overlap

        # Count technical indicators
        technical_indicators = [
            r"::",
            r"\(\)",
            r"<.*>",
            r"template",
            r"const",
            r"virtual",
            r"boost",
            r"std",
            r"namespace",
            r"class",
            r"struct",
        ]

        technical_count = sum(
            len(re.findall(pattern, section_text, re.IGNORECASE))
            for pattern in technical_indicators
        )

        density = (technical_count / len(words)) * 100

        # Adjust overlap based on density
        if density > 5:  # High density
            return int(base_overlap * 1.5)
        elif density > 2:  # Medium density
            return int(base_overlap * 1.2)
        else:
            return base_overlap


class SemanticChunker:
    """Semantic chunker that preserves document structure and creates meaningful chunks."""

    def __init__(self, embedding_model=None):
        """
        Initialize semantic chunker with shared models.

        Args:
            model_name: Name of sentence transformer model for semantic similarity (deprecated, use embedding_model)
            language: Language setting
            embedding_model: Shared embedding model instance
            text_generation_model: Shared text generation model instance
        """
        self.logger = logger.bind(name="SemanticChunker")
        self.model_name = get_config(
            "rag.embedding.minilm.model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        # Use shared embedding model if provided
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        # if embedding_model:
        #     self.model = embedding_model.model
        #     self.model_name = embedding_model.model_name
        #     self.logger.info("✅ Using shared embedding model for semantic chunking")
        # else:
        #     self.logger.error("No shared embedding model provided, using default model")
        #     return False

        self.classifier = pipeline(
            "zero-shot-classification", model="cross-encoder/nli-distilroberta-base"
        )

        # Chunking parameters
        self.min_chunk_size = get_config("rag.semantic_chunking.min_chunk_size", 100)
        self.max_chunk_size = get_config("rag.semantic_chunking.max_chunk_size", 1000)
        self.similarity_threshold = get_config(
            "rag.semantic_chunking.semantic_similarity_threshold", 0.7
        )
        self.overlap_size = get_config("rag.chunk_overlap", 50)

        # Dynamic windowing parameters
        self.dynamic_window_enabled = get_config(
            "rag.semantic_chunking.dynamic_window_enabled", True
        )
        self.base_chunk_size = get_config("rag.semantic_chunking.base_chunk_size", 512)
        self.complexity_multiplier = get_config(
            "rag.semantic_chunking.complexity_multiplier", 1.5
        )
        self.similarity_adaptive = get_config(
            "rag.semantic_chunking.similarity_adaptive", True
        )
        self.content_type_weights = {
            "code": 1.2,  # Code blocks need more space
            "api_reference": 1.1,  # API docs are dense
            "header": 0.8,  # Headers are short
            "text": 1.0,  # Regular text
            "example": 1.3,  # Examples need more context
        }

        # Boost.Asio specific patterns
        self.code_patterns = {
            "cpp_function": r"^\s*(?:template\s*<[^>]*>\s*)?(?:inline\s+)?(?:static\s+)?(?:const\s+)?(?:volatile\s+)?(?:explicit\s+)?(?:virtual\s+)?(?:friend\s+)?(?:constexpr\s+)?(?:noexcept\s*\([^)]*\)\s*)?(?:override\s+)?(?:final\s+)?(?:class\s+|struct\s+|enum\s+|union\s+)?\w+(?:\s*<[^>]*>)?\s+(?:\*|\&)?\s*\w+\s*\([^)]*\)\s*(?:const\s*)?(?:\s*=\s*(?:0|default|delete))?\s*[;{]",
            "cpp_include": r'^\s*#\s*include\s*[<"][^>"]*[>"]',
            "cpp_define": r"^\s*#\s*define\s+\w+",
            "cpp_namespace": r"^\s*namespace\s+\w+",
            "cpp_class": r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+\w+",
            "cpp_comment": r"^\s*//.*$",
            "cpp_block_comment": r"/\*.*?\*/",
        }

        self.markdown_patterns = {
            "header": r"^#{1,6}\s+.+",
            "code_block": r"```[\s\S]*?```",
            "inline_code": r"`[^`]+`",
            "list_item": r"^\s*[-*+]\s+",
            "numbered_list": r"^\s*\d+\.\s+",
            "table_row": r"^\s*\|.*\|",
            "link": r"\[([^\]]+)\]\(([^)]+)\)",
        }

        self.chunk_types = ["code", "header", "text", "example", "api_reference"]

        if embedding_model:
            self.logger.info(
                f"✅ SemanticChunker initialized with shared embedding model: {self.model_name}"
            )
        else:
            self.logger.info(
                f"SemanticChunker initialized with model: {self.model_name}"
            )

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
                if "function" in pattern_name or "class" in pattern_name:
                    return "api_reference"
                return "code"

        # Check for markdown patterns
        if re.match(self.markdown_patterns["header"], text):
            return "header"

        if re.search(self.markdown_patterns["code_block"], text):
            return "code"

        if re.search(self.markdown_patterns["inline_code"], text):
            return "api_reference"

        # Check for example indicators
        example_indicators = ["example", "demo", "sample", "usage", "tutorial"]
        if any(indicator in text.lower() for indicator in example_indicators):
            return "example"

        return "text"

    def extract_function_name(self, text: str) -> Optional[str]:
        """Extract function name from C++ code."""
        # Simple function name extraction
        func_match = re.search(r"(\w+)\s*\([^)]*\)", text)
        if func_match:
            return func_match.group(1)
        return None

    def split_by_structure(
        self, text: str, source_file: str
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Split text by structural elements (headers, code blocks, etc.).

        Args:
            text: Text to split
            source_file: Source file path

        Returns:
            List of (chunk_text, metadata) tuples
        """
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_metadata = None
        line_number = 0

        for i, line in enumerate(lines):
            line_number = i + 1
            content_type = self.detect_content_type(line)

            # Start new chunk if content type changes significantly
            if (
                current_metadata
                and current_metadata.chunk_type != content_type
                and current_chunk
                and len("\n".join(current_chunk)) > self.min_chunk_size
            ):

                # Finalize current chunk
                chunk_text = "\n".join(current_chunk)
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
                    function_name=(
                        self.extract_function_name(line)
                        if content_type == "api_reference"
                        else None
                    ),
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
                        function_name=(
                            self.extract_function_name(line)
                            if content_type == "api_reference"
                            else None
                        ),
                    )

                current_chunk.append(line)
                current_metadata.end_line = line_number

        # Add final chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
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
        except (ValueError, RuntimeError, TypeError) as e:
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
        code_elements = len(re.findall(r"```[\s\S]*?```", text))
        inline_code = len(re.findall(r"`[^`]+`", text))
        functions = len(re.findall(r"\w+\s*\([^)]*\)", text))
        classes = len(re.findall(r"(?:class|struct)\s+\w+", text))

        # Count structural elements
        headers = len(re.findall(r"^#{1,6}\s+", text, re.MULTILINE))
        lists = len(re.findall(r"^\s*[-*+]\s+", text, re.MULTILINE))
        tables = len(re.findall(r"^\s*\|.*\|", text, re.MULTILINE))

        # Calculate complexity based on content density
        total_elements = (
            code_elements + inline_code + functions + classes + headers + lists + tables
        )
        text_length = len(text)

        if text_length > 0:
            element_density = total_elements / (
                text_length / 100
            )  # Elements per 100 chars

            if element_density > 2.0:
                complexity_score = 1.8  # Very complex
            elif element_density > 1.5:
                complexity_score = 1.5  # Complex
            elif element_density > 1.0:
                complexity_score = 1.2  # Moderately complex
            elif element_density < 0.3:
                complexity_score = 0.7  # Simple

        return max(0.5, min(2.0, complexity_score))

    def calculate_dynamic_chunk_size(
        self, content_type: str, complexity: float, previous_similarity: float = 0.5
    ) -> int:
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

    def calculate_dynamic_similarity_threshold(self, content_type: str) -> float:
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
        if content_type == "code":
            # Code blocks should be more strictly similar
            return min(0.9, base_threshold + 0.1)
        elif content_type == "header":
            # Headers can be less similar
            return max(0.5, base_threshold - 0.1)
        elif content_type == "api_reference":
            # API references need moderate similarity
            return base_threshold
        else:
            return base_threshold

    def merge_similar_chunks(
        self, chunks: List[Tuple[str, ChunkMetadata]]
    ) -> List[Tuple[str, ChunkMetadata]]:
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
                current_metadata.chunk_type, current_complexity, previous_similarity
            )
            dynamic_threshold = self.calculate_dynamic_similarity_threshold(
                current_metadata.chunk_type
            )

            # Calculate similarity with next chunk
            similarity = self.calculate_semantic_similarity(
                current_chunk_text, next_chunk_text
            )

            # Check if chunks should be merged using dynamic criteria
            should_merge = False

            # Merge if current chunk is smaller than dynamic size
            if len(current_chunk_text) < dynamic_chunk_size:
                should_merge = True

            # Merge if chunks are semantically similar above dynamic threshold
            elif (
                current_metadata.chunk_type == next_metadata.chunk_type
                and similarity > dynamic_threshold
            ):
                should_merge = True

            # Merge if both chunks are small relative to their dynamic sizes
            elif (
                len(current_chunk_text) < dynamic_chunk_size * 1.2
                and len(next_chunk_text) < dynamic_chunk_size * 1.2
                and current_metadata.chunk_type == next_metadata.chunk_type
            ):
                should_merge = True

            if should_merge:
                # Merge chunks
                current_chunk_text += "\n" + next_chunk_text
                current_metadata.end_line = next_metadata.end_line

                # Update chunk ID to reflect merged nature
                current_metadata.chunk_id = (
                    f"{current_metadata.chunk_id}_{next_metadata.chunk_id}"
                )

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

    def split_large_chunks(
        self, chunks: List[Tuple[str, ChunkMetadata]]
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Split chunks that are too large using dynamic windowing.

        Args:
            chunks: List of (chunk_text, metadata) tuples

        Returns:
            Split chunks with dynamic sizing
        """
        split_chunks = []
        start_line = 1
        for chunk_text, metadata in chunks:
            # Calculate dynamic max size for this content type
            if len(chunk_text) < self.max_chunk_size:
                metadata.start_line = start_line
                start_line += len(chunk_text.split("\n"))
                metadata.end_line = start_line - 1
                split_chunks.append((chunk_text, metadata))
                continue

            complexity = self.calculate_content_complexity(chunk_text)
            dynamic_max_size = self.calculate_dynamic_chunk_size(
                metadata.chunk_type, complexity, 0.5  # Default similarity
            )

            # Split large chunks by sentences or logical breaks
            lines = chunk_text.split("\n")
            current_chunk_lines = []
            current_length = 0
            chunk_index = 0

            for line in lines:
                line_length = len(line) + 1  # +1 for newline

                if (
                    current_length + line_length > dynamic_max_size
                    and current_chunk_lines
                ):
                    # Create new chunk

                    new_chunk_text = "\n".join(current_chunk_lines)
                    new_metadata = ChunkMetadata(
                        chunk_id=f"{metadata.chunk_id}_part_{chunk_index}",
                        source_file=metadata.source_file,
                        chunk_type=metadata.chunk_type,
                        start_line=start_line,
                        end_line=start_line + len(current_chunk_lines) - 1,
                        parent_section=metadata.parent_section,
                        function_name=metadata.function_name,
                        example_type=metadata.example_type,
                        importance_score=metadata.importance_score,
                    )
                    split_chunks.append((new_chunk_text, new_metadata))

                    # Start new chunk with dynamic overlap
                    overlap_size = min(
                        self.overlap_size, dynamic_max_size // 10
                    )  # 10% overlap
                    overlap_lines = (
                        current_chunk_lines[-overlap_size // 50 :]
                        if overlap_size > 0
                        else []
                    )
                    current_chunk_lines = overlap_lines + [line]
                    current_length = sum(len(l) + 1 for l in current_chunk_lines)
                    chunk_index += 1
                    start_line += len(current_chunk_lines)
                else:
                    current_chunk_lines.append(line)
                    current_length += line_length

            # Add final chunk
            if current_chunk_lines:
                new_chunk_text = "\n".join(current_chunk_lines)
                new_metadata = ChunkMetadata(
                    chunk_id=f"{metadata.chunk_id}_part_{chunk_index}",
                    source_file=metadata.source_file,
                    chunk_type=metadata.chunk_type,
                    start_line=start_line,
                    end_line=start_line + len(current_chunk_lines) - 1,
                    parent_section=metadata.parent_section,
                    function_name=metadata.function_name,
                    example_type=metadata.example_type,
                    importance_score=metadata.importance_score,
                )
                split_chunks.append((new_chunk_text, new_metadata))

        return split_chunks

    def classify_text(self, text):
        return self.classifier(text, candidate_labels=self.chunk_types)

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
            with ThreadPoolExecutor(max_workers=4) as executor:
                chunk_types = [
                    executor.submit(self.classify_text, chunk[0])
                    for chunk in final_chunks
                ]
                start_line = 0
                for i, chunk_type in enumerate(as_completed(chunk_types)):
                    chunk_text = final_chunks[i][0]
                    metadata = final_chunks[i][1]
                    chunk_id_hash = hashlib.md5(
                        metadata.chunk_id.encode(), usedforsecurity=False
                    ).hexdigest()

                    chunk_dict = {
                        "chunk_id": chunk_id_hash,
                        "text": chunk_text,
                        "metadata": {
                            "source_file": source_file,
                            "chunk_type": chunk_type.result()["labels"][0],
                            "start_line": start_line,
                            "end_line": metadata.end_line,
                            "parent_section": metadata.parent_section,
                            "function_name": metadata.function_name,
                            "example_type": metadata.example_type,
                            "importance_score": metadata.importance_score,
                            "chunk_index": i,
                            "chunk_length": len(chunk_text),
                        },
                    }
                    start_line = metadata.end_line + 1
                    chunk_dicts.append(chunk_dict)

            return chunk_dicts

        except Exception as e:
            self.logger.error(f"Error chunking document {source_file}: {e}")
            return []


class ChunkerProcessor:
    """Process documents into semantic chunks using heading-aware and semantic chunking strategies."""

    def __init__(
        self,
        language: str = None,
        file_processor: MultiFormatProcessor = None,
        embedding_model=None,
    ):
        self.logger = logger.bind(name="ChunkerProcessor")
        self.language = language
        if file_processor is None:
            self.file_processor = MultiFormatProcessor(language=self.language)
        else:
            self.file_processor = file_processor
        # self.old_chunker = SemanticChunker(embedding_model=embedding_model)
        self.chunker = HeadingAwareChunker()
        self.boilerplate_remover = BoilerplateRemover()

    def get_write_path(
        self, file_path: Path, output_path: Path, input_path: Path
    ) -> Path:
        """Generate output path for chunked file based on input file location."""
        # use the same relative path as the input file
        if input_path in file_path.parents:
            new_file_path = (
                str(file_path.relative_to(input_path)).split(".")[0]
                + "_semantic_chunks.json"
            )
            return output_path / new_file_path
        else:
            return output_path / f"{file_path.stem}_semantic_chunks.json"

    def get_source_url(self, text: str, file_path: Path) -> str:
        if "Source URL: " in text.split("\n")[0]:
            return text.split("\n")[0].replace("Source URL: ", "")

        if "git_source" in file_path.as_posix():

            library_name = file_path.as_posix().split("/")[0]
            given_url = file_path.as_posix().replace(f"{library_name}/git_source/", "")
            source_url = f"https://github.com/boostorg/{library_name}/blob/boost-1.89.0/{given_url}"
            return source_url

    def process_knowledge_base(
        self,
        raw_file_list: List[Path] = None,
        max_files: int = None,
        input_dir: str = None,
    ) -> Tuple[bool, List[str]]:
        """
        Process entire knowledge base with semantic chunking.

        Args:
            raw_file_list: List of raw file paths

        Returns:
            True if successful, False otherwise
        """
        try:
            # Allow caller to override the input directory (e.g., use processed path)
            if input_dir is not None:
                input_path = Path(input_dir)
            else:
                input_dir = f"{get_config('data.source_data.processed_data_path', 'data/source_data/processed')}/{self.language}"
                input_path = Path(input_dir)
            if raw_file_list is None:
                raw_file_list = self.file_processor.get_file_list(input_path)
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
                write_path = self.get_write_path(file_path, output_path, input_path)

                text = self.file_processor.extract_text_from_file(file_path)
                source_url = self.get_source_url(
                    text, Path(file_path).relative_to(input_path)
                )
                metadata = {"source_url": source_url}
                # remove extra newlines
                text = re.sub(r"\n\s*\n+", "\n\n", text)
                lines = [line.rstrip() for line in text.split("\n")]
                text = "\n".join(lines)

                if text != "":
                    self.logger.info(f"Processing: {file_path}")
                    if write_path.exists():
                        self.logger.info(
                            f"Skipping {write_path} because it already exists"
                        )
                        # continue
                    chunks = self.chunker.chunk_document(text, metadata)
                    # old_chunks = self.old_chunker.chunk_document(text, str(file_path))
                    chunks = self.boilerplate_remover.clean_chunks(chunks)
                    # old_chunks = self.boilerplate_remover.clean_chunks(old_chunks)
                    if chunks:
                        # Save chunks
                        write_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(write_path, "w", encoding="utf-8") as f:
                            json.dump(chunks, f, ensure_ascii=False, indent=2)
                        chunk_file_list.append(write_path)
                        total_chunks += len(chunks)
                        processed_files += 1

            self.logger.info(
                f"Processed {processed_files} files, created {total_chunks} semantic chunks"
            )
            return True, chunk_file_list

        except Exception as e:
            self.logger.error(f"Error processing knowledge base: {e}")
            return False, chunk_file_list


def main():
    """Main function for command-line usage."""
    # input_dir = "data/source_data/processed/cn/"
    # output_dir = get_config("rag.semantic_chunks_dir", "data/processed/semantic_chunks")
    # model_name = get_config("rag.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    chunker = ChunkerProcessor(language="en")

    if chunker.process_knowledge_base(input_dir=None):
        print("Semantic chunking completed successfully!")
    else:
        print("Semantic chunking failed!")
        exit(1)


if __name__ == "__main__":
    main()
