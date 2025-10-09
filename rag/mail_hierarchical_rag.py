"""
Mail hierarchy graph built from processed mail JSON records (parent/children).
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import networkx as nx
import faiss
from enum import Enum
import pickle

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import get_config, RetrievalResult, get_model_nick_name
from data_processor.summarize_processor import SummarizePocessor

class HierarchyLevel(Enum):
    """Hierarchy levels for document structure."""
    LIBRARY = "library"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    CONVERSATION = "conversation"
    THREAD = "thread"
    MESSAGE = "message"


@dataclass
class HierarchicalNode:
    """Node in hierarchical structure."""
    node_id: str
    level: HierarchyLevel
    content: str
    summary: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}

class MailHierarchicalRAG:
    """Mail hierarchy graph built from processed mail JSON records (parent/children).

    Responsibilities:
    - Discover processed mail files
    - Build graph from files with pre-computed embeddings and summaries
    - Update existing graph with new files
    - Save/load graph to/from disk (NetworkX and Neo4j)
    - Fast semantic search using pre-computed embeddings
    """

    def __init__(self, language: str = None, embedder=None, summarizer=None):
        self.logger = logger.bind(name="MailHierarchicalRAG")
        if language is not None:
            self.language = language
        self.graph = nx.DiGraph()
        
        # Initialize components from boost pipeline or create new ones
        self.embedder = embedder
        self.summarizer = summarizer
        self.vector_index = None  # FAISS index for fast similarity search
        self.node_embeddings = {}  # Pre-computed embeddings
        self.node_summaries = {}   # Pre-computed summaries
        self.index_to_node = None  # List from index to node_id
        self.graph_filename = get_config("data.processed_data.message_by_thread_cache_filename", "message_by_thread.pkl")

        self.file_path = Path(get_config("data.processed_data.message_by_thread_cache_path", "data/processed/message_by_thread_cache"))
        self.file_path = self.file_path / get_model_nick_name(self.embedder.model_card_data.base_model)
        self.file_path = self.file_path / self.language
        self.graph_full_path = self.file_path / self.graph_filename

    def _discover_files(self, path_or_list: Any = None) -> List[Path]:
        """Discover processed mail files from message_by_thread directory."""
        
        if isinstance(path_or_list, list) and len(path_or_list) > 0:
            return path_or_list
        elif isinstance(path_or_list, str):
            if path_or_list.endswith(".json"):
                return [Path(path_or_list)]
            elif path_or_list.split("/")[-1] == self.language:
                thread_dir = Path(path_or_list)
            else:
                thread_dir = Path(path_or_list) / self.language
        else:
            thread_dir = Path(get_config("data.processed_data.message_by_thread_path", "data/processed/message_by_thread")) / self.language
        
        # Look for thread files in message_by_thread directory
        if not thread_dir.exists():
            self.logger.info(f"No message_by_thread directory found: {thread_dir}")
            return []
        # Find all thread JSON files
        files = list(thread_dir.rglob("*_thread_*.json"))
        
        self.logger.info(f"Discovered {len(files)} thread files")
        return files

    def _load_records(self, files: List[Path]) -> Dict[str, Dict[str, Any]]:
        """Load records from thread files and return a dict keyed by message_id."""
        id_to_record: Dict[str, Dict[str, Any]] = {}
        total_messages = 0
        processed_files = 0
        
        for i, pf in enumerate(files, 1):
            try:
                # Log file loading progress
                if i % 100 == 0:
                    file_progress = (i / len(files)) * 100
                    self.logger.info(f"  📄 Loading file {i}/{len(files)}: ({file_progress:.1f}%)")
                
                messages = json.loads(Path(pf).read_text(encoding="utf-8"))
                thread_name = None
                if not isinstance(messages, list):
                    thread_name = messages.get('thread_info', {}).get('url')
                    messages = messages.get('messages', [])
                
                file_messages = 0
                for message in messages:
                    msg_id = message.get("message_id")
                    if not msg_id:
                        continue
                    
                    # Create a hash of message_id for consistency
                    import hashlib
                    msg_id_hash = hashlib.md5(msg_id.encode()).hexdigest()
                    
                    # Store the message with additional metadata
                    id_to_record[msg_id_hash] = {
                        "message_id": msg_id,
                        "message_id_hash": msg_id_hash,
                        "content": message,
                        "thread_file": str(Path(pf).stem),
                        "thread_name": thread_name if thread_name else message.get("thread_url")
                    }
                    total_messages += 1
                    file_messages += 1
                
                processed_files += 1
                    
            except Exception as exc:
                self.logger.warning(f"Failed to load {pf}: {exc}")
                continue
        
        self.logger.info(f"📊 Loaded {total_messages} messages from {processed_files}/{len(files)} thread files")
        return id_to_record

   
    def build_graph(self, processed_files: List[Path] = None) -> bool:
        """Build a new mail hierarchy graph with pre-computed embeddings and summaries."""
        self.logger.info("🚀 Starting MailHierarchicalGraph build process...")
        
        # Step 1: Discover files
        self.logger.info("📁 Step 1/5: Discovering thread files...")
        files = self._discover_files(processed_files)
        if not files:
            self.logger.warning("❌ No thread files found to build graph")
            return False
        self.logger.info(f"✅ Step 1/5 Complete: Found {len(files)} thread files (100.0%)")
        
        folder_name = Path(files[0]).parent.name
        if folder_name and folder_name != self.language:
            self.graph_full_path = self.file_path / folder_name / self.graph_filename

        
        # Step 2: Load records
        self.logger.info("📄 Step 2/5: Loading message records...")
        self.graph = nx.DiGraph()
        id_to_record = self._load_records(files)
        
        if not id_to_record:
            self.logger.warning("❌ No messages loaded from files")
            return False
        self.logger.info(f"✅ Step 2/5 Complete: Loaded {len(id_to_record)} messages (100.0%)")

        # Step 3: Process messages with summarization and embedding
        self.logger.info("🔄 Step 3/5: Processing messages with summarization and embedding...")
        processed_count = self.summarize_and_embed_messages(id_to_record)
        self.logger.info(f"✅ Step 3/5 Complete: Processed {processed_count} messages (100.0%)")


        # Final summary
        self.logger.info("🎉 MailHierarchicalGraph build completed successfully!")
        self.logger.info(f"📊 Final Statistics:")
        self.logger.info(f"  - Nodes: {self.graph.number_of_nodes()}")
        self.logger.info(f"  - Edges: {self.graph.number_of_edges()}")
        self.logger.info(f"  - Pre-computed summaries: {len(self.node_summaries)}")
        self.logger.info(f"  - Pre-computed embeddings: {len(self.node_embeddings)}")
        self.logger.info(f"  - Vector index: {'✅ Built' if self.vector_index else '❌ Failed'}")
        
        return True
    
    def summarize_and_embed_messages(self, id_to_record: Dict[str, Dict[str, Any]], is_update: bool = False) -> bool:
        """Summarize and embed messages."""
        processed_count = 0
        total_messages = len(id_to_record)
        new_records = {}
        for msg_id_hash, record in id_to_record.items():
            try:
                if msg_id_hash in self.graph and not is_update:
                    continue
                content = record["content"]
                msg_id = record["message_id"]
                
                # Create text content for summarization and embedding
                text_content = self._create_message_text(content)
                
                # Generate embedding
                embedding, summary = self._generate_embedding(text_content)
                
                # Store pre-computed data
                self.node_summaries[msg_id_hash] = summary
                self.node_embeddings[msg_id_hash] = embedding
                
                # Add node to graph with all attributes
                attrs = {
                    "message_id": msg_id,
                    "subject": content.get("subject", ""),
                    "sender_address": content.get("sender_address", ""),
                    "from": content.get("from", ""),
                    "date": content.get("date", ""),
                    "to": content.get("to", ""),
                    "cc": content.get("cc", ""),
                    "reply_to": content.get("reply_to", ""),
                    "url": content.get("url", ""),
                    "summary": summary,
                    "thread_name": record["thread_name"],
                    "thread_file": record["thread_file"],
                    "content_length": len(text_content)
                }
                self.graph.add_node(msg_id_hash, **attrs)
                new_records[msg_id_hash] = record
                
                processed_count += 1
                
                # Log progress every 100 messages or at completion
                if processed_count % 1000 == 0 or processed_count == total_messages:
                    progress = (processed_count / total_messages) * 100
                    self.logger.info(f"  📊 Processing progress: {progress:.1f}% ({processed_count}/{total_messages})")
                    edge_count = self._build_relationships(new_records)
                    self.logger.info(f"✅ Built {edge_count} relationships")
                    self._build_vector_index()
                    self.save_graph()
                    new_records = {}
                    
            except Exception as e:
                self.logger.warning(f"Failed to process message {msg_id}: {e}")
                continue
        return processed_count

    def _build_relationships(self, id_to_record: Dict[str, Dict[str, Any]]) -> bool:
        edge_count = 0
        total_relationships = sum(1 for record in id_to_record.values() 
                                for _ in [record["content"].get("parent")] + record["content"].get("children", []) 
                                if _)
        processed_relationships = 0
        
        for msg_id_hash, record in id_to_record.items():
            content = record["content"]
            parent = content.get("parent")
            children = content.get("children", [])
            
            # Add parent relationship
            if parent:
                processed_relationships += 1
                parent_hash = self._find_message_hash_by_id(parent, id_to_record)
                if parent_hash and parent_hash in self.graph:
                    self.graph.add_edge(parent_hash, msg_id_hash, relationship="parent")
                    edge_count += 1
            
            # Add child relationships
            for child in children:
                processed_relationships += 1
                child_hash = self._find_message_hash_by_id(child, id_to_record)
                if child_hash and child_hash in self.graph:
                    self.graph.add_edge(msg_id_hash, child_hash, relationship="child")
                    edge_count += 1
            
            # Log progress every 100 relationships
            if processed_relationships % 100 == 0 or processed_relationships == total_relationships:
                progress = (processed_relationships / total_relationships) * 100 if total_relationships > 0 else 100
                self.logger.info(f"  📊 Relationship progress: {progress:.1f}% ({processed_relationships}/{total_relationships})")
    
        return edge_count
    
    
    def _create_message_text(self, content: Dict[str, Any]) -> str:
        """Create searchable text from message content."""
        text_parts = []
        
        # Add subject
        if content.get("subject"):
            text_parts.append(f"subject: {content['subject']}")
        
        # Add content
        if content.get("content"):
            text_parts.append(f"content: {content['content']}")
        
        # Add sender info
        if content.get("from"):
            text_parts.append(f"sender: {content['from']}")
        
        return "\n".join(text_parts)

    def _generate_summary(self, text: str) -> str:
        """Generate summary using the summarizer."""
        try:
            if not text or len(text.strip()) < 50:
                return text[:200] if text else ""
            
            summary = self.summarizer.summarize_2_3_sentences(text)
            return summary if summary else text[:200]
            
        except Exception as e:
            self.logger.warning(f"Failed to generate summary: {e}")
            return text[:200] if text else ""

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using the embedding model."""
        try:
            summary = text if text else ""
            if not text or len(text.strip()) < 10:
                # Return zero vector for empty text
                return np.zeros(768), summary  # Assuming 768-dimensional embeddings
            
            # Use the embedding model to generate embedding
            try:
                embedding = self.embedder.encode(text)
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding: {e}, using summary")
                summary = self._generate_summary(text)
                embedding = self.embedder.encode(summary)
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding: {e}, using text")
                return np.zeros(768), summary
            
            return embedding, summary
            
        except Exception as e:
            self.logger.warning(f"Failed to generate embedding: {e}")
            return np.zeros(768), summary

    def _find_message_hash_by_id(self, message_id: str, id_to_record: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Find message hash by message_id."""
        for msg_hash, record in id_to_record.items():
            if record["content"]["url"] == message_id:
                return msg_hash
        return None

    def _build_vector_index(self):
        """Build FAISS index for fast similarity search."""
        try:
            if not self.node_embeddings:
                self.logger.warning("No embeddings available for vector index")
                return
            
            self.logger.info("  🔄 Converting embeddings to numpy array...")
            # Convert embeddings to numpy array
            embeddings = []
            node_ids = []
            total_embeddings = len(self.node_embeddings)
            
            for i, (node_id, embedding) in enumerate(self.node_embeddings.items(), 1):
                embeddings.append(embedding)
                node_ids.append(node_id)
                
                # Log progress every 1000 embeddings
                if i % 1000 == 0 or i == total_embeddings:
                    progress = (i / total_embeddings) * 100
                    self.logger.info(f"    📊 Embedding conversion: {progress:.1f}% ({i}/{total_embeddings})")
            
            self.logger.info("  🔄 Creating FAISS index...")
            embeddings = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            self.logger.info("  🔄 Normalizing embeddings for cosine similarity...")
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            self.logger.info("  🔄 Adding embeddings to FAISS index...")
            self.vector_index.add(embeddings)
            
            # Store mapping from index to node_id
            self.index_to_node = node_ids
            
            self.logger.info(f"  ✅ Built FAISS index with {len(embeddings)} vectors of dimension {dimension}")
            
        except Exception as e:
            self.logger.error(f"Failed to build vector index: {e}")
            self.vector_index = None

    # -----------------------------
    # Helper utilities to reduce duplication
    # -----------------------------
    def _threshold_by_top(self, results: List[RetrievalResult], ratio: float) -> List[RetrievalResult]:
        """Filter results keeping those with score >= ratio * top_score. Returns new list."""
        if not results:
            return []
        try:
            top_score = max(r.score for r in results)
            threshold = top_score * ratio
            return [r for r in results if r.score >= threshold]
        except Exception:
            return results

    def _apply_recency(self, results: List[RetrievalResult], *, alpha: float = 0.7, beta: float = 0.3,
                        accumulate: bool = False, prev_scale: float = 1.0, add_scale: float = 1.0) -> None:
        """Apply recency-aware reweighting.
        - alpha/beta control blend of relevance and recency: score * (alpha + beta * recency_weight)
        - if accumulate, mix with existing gen_score using prev_scale; then add add_scale * new contribution
        Writes to r.gen_score.
        """
        for r in results:
            try:
                rec = self._compute_recency_weight(r.metadata)
                base = (getattr(r, 'gen_score', 0.0) * prev_scale) if accumulate else 0.0
                r.gen_score = base + (r.score * (alpha + beta * rec) * add_scale)
            except Exception:
                # Fallback: keep previous gen_score if exists, else mirror raw score
                if not hasattr(r, 'gen_score'):
                    r.gen_score = r.score

    def _normalize_vector(self, vec: Any) -> Optional[np.ndarray]:
        """Convert to float32 numpy array and L2-normalize. Return None if invalid or zero-norm."""
        try:
            if vec is None:
                return None
            if isinstance(vec, list):
                arr = np.array(vec, dtype=np.float32)
            elif isinstance(vec, np.ndarray):
                arr = vec.astype('float32', copy=False)
            else:
                # Unknown type (e.g., torch.Tensor) - attempt numpy conversion
                arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm == 0 or not np.isfinite(norm):
                return None
            return arr / norm
        except Exception:
            return None

    def _get_normalized_node_vector(self, node_id: str) -> Optional[np.ndarray]:
        """Fetch node embedding and return L2-normalized vector, or None if missing/invalid."""
        emb = self.node_embeddings.get(node_id)
        return self._normalize_vector(emb)

    def _finalize_results(self, results: List[RetrievalResult], *,
                           threshold_ratio: Optional[float] = None,
                           recency: bool = True,
                           accumulate: bool = False,
                           prev_scale: float = 1.0,
                           add_scale: float = 1.0,
                           do_sort: bool = False,
                           limit: Optional[int] = None) -> List[RetrievalResult]:
        """Shared tail-processing for search results.
        - Optionally threshold by top score.
        - Optionally apply recency reweighting (writes gen_score).
        - Optionally sort by gen_score (fallback to score if missing).
        - Optionally slice to limit.
        """
        if not results:
            return []
        processed = results
        if threshold_ratio is not None:
            processed = self._threshold_by_top(processed, ratio=threshold_ratio)
        if recency:
            self._apply_recency(processed, alpha=0.7, beta=0.3,
                                accumulate=accumulate, prev_scale=prev_scale, add_scale=add_scale)
        if do_sort:
            processed.sort(key=lambda x: getattr(x, 'gen_score', getattr(x, 'score', 0.0)), reverse=True)
        if limit is not None:
            return processed[:limit]
        return processed

    def update_message_by_node(self, node_id: str, content: Dict[str, Any]) -> bool:
        """Update an existing message node in the graph."""
        try:
            if not self.graph or node_id not in self.graph:
                self.logger.warning(f"Node {node_id} not found in graph")
                return False
            
            self.logger.info(f"🔄 Updating message node: {node_id}")
            
            # Create text content for summarization and embedding
            text_content = self._create_message_text(content)
            
            # Generate new embedding and summary
            embedding, summary = self._generate_embedding(text_content)
            
            # Update pre-computed data
            self.node_summaries[node_id] = summary
            self.node_embeddings[node_id] = embedding
            
            # Update node attributes
            attrs = {
                "subject": content.get("subject", ""),
                "sender_address": content.get("sender_address", ""),
                "from": content.get("from", ""),
                "date": content.get("date", ""),
                "to": content.get("to", ""),
                "cc": content.get("cc", ""),
                "reply_to": content.get("reply_to", ""),
                "url": content.get("url", ""),
                "summary": summary,
                "content_length": len(text_content)
            }
            
            # Update node attributes
            for key, value in attrs.items():
                self.graph.nodes[node_id][key] = value
            
            # Rebuild vector index if it exists
            if self.vector_index:
                self.logger.info("🔄 Rebuilding vector index after update...")
                self._build_vector_index()
            
            self.logger.info(f"✅ Successfully updated message node: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update message node {node_id}: {e}")
            return False
    
    def add_new_message(self, message: Dict[str, Any]) -> bool:
        """Add a new message to the graph."""
        try:
            if not self.graph:
                self.logger.warning("Graph not initialized")
                return False
            
            # Generate unique node ID
            import hashlib
            msg_id = message.get("message_id", "")
            if not msg_id:
                self.logger.warning("Message ID is required")
                return False
            
            msg_id_hash = hashlib.md5(msg_id.encode()).hexdigest()
            
            # Check if message already exists
            if msg_id_hash in self.graph:
                self.logger.warning(f"Message {msg_id_hash} already exists, updating instead")
                return self.update_message_by_node(msg_id_hash, message)
            
            self.logger.info(f"➕ Adding new message: {msg_id_hash}")
            
            # Create text content for summarization and embedding
            text_content = self._create_message_text(message)
            
            # Generate embedding and summary
            embedding, summary = self._generate_embedding(text_content)
            
            # Store pre-computed data
            self.node_summaries[msg_id_hash] = summary
            self.node_embeddings[msg_id_hash] = embedding
            
            # Add node to graph with all attributes
            attrs = {
                "message_id": msg_id,
                "subject": message.get("subject", ""),
                "sender_address": message.get("sender_address", ""),
                "from": message.get("from", ""),
                "date": message.get("date", ""),
                "to": message.get("to", ""),
                "cc": message.get("cc", ""),
                "reply_to": message.get("reply_to", ""),
                "url": message.get("url", ""),
                "summary": summary,
                "thread_name": message.get("thread_id", ""),
                "thread_file": message.get("thread_file", ""),
                "content_length": len(text_content)
            }
            self.graph.add_node(msg_id_hash, **attrs)
            
            # Add relationships if parent/children exist
            parent = message.get("parent")
            children = message.get("children", [])
            
            # Add parent relationship
            if parent:
                parent_hash = self._find_message_hash_by_id(parent, {})
                if parent_hash and parent_hash in self.graph:
                    self.graph.add_edge(parent_hash, msg_id_hash, relationship="parent")
                    self.logger.info(f"🔗 Added parent relationship: {parent_hash} -> {msg_id_hash}")
            
            # Add child relationships
            for child in children:
                child_hash = self._find_message_hash_by_id(child, {})
                if child_hash and child_hash in self.graph:
                    self.graph.add_edge(msg_id_hash, child_hash, relationship="child")
                    self.logger.info(f"🔗 Added child relationship: {msg_id_hash} -> {child_hash}")
            
            # Rebuild vector index to include new message
            if self.vector_index:
                self.logger.info("🔄 Rebuilding vector index with new message...")
                self._build_vector_index()
            
            self.logger.info(f"✅ Successfully added new message: {msg_id_hash}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add new message: {e}")
            return False
    
    def add_thread(self, thread_data: Dict[str, Any]) -> bool:
        """Add a new thread to the graph."""
        try:
            if not self.graph:
                self.logger.warning("Graph not initialized")
                return False
            
            thread_id = thread_data.get("thread_id", "")
            if not thread_id:
                self.logger.warning("Thread ID is required")
                return False
            
            # Check if thread already exists
            if thread_id in self.graph:
                self.logger.warning(f"Thread {thread_id} already exists")
                return False
            
            self.logger.info(f"➕ Adding new thread: {thread_id}")
            
            # Add thread node to graph
            attrs = {
                "thread_id": thread_id,
                "url": thread_data.get("url", ""),
                "subject": thread_data.get("subject", ""),
                "date_active": thread_data.get("date_active", ""),
                "starting_email": thread_data.get("starting_email", ""),
                "emails_url": thread_data.get("emails_url", ""),
                "replies_count": thread_data.get("replies_count", 0),
                "votes_total": thread_data.get("votes_total", 0),
                "node_type": "thread"
            }
            self.graph.add_node(thread_id, **attrs)
            
            self.logger.info(f"✅ Successfully added new thread: {thread_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add new thread: {e}")
            return False
    
    def add_message_to_thread(self, thread_id: str, message_data: Dict[str, Any]) -> bool:
        """Add a message to a specific thread."""
        try:
            if not self.graph:
                self.logger.warning("Graph not initialized")
                return False
            
            # Check if thread exists
            if thread_id not in self.graph:
                self.logger.warning(f"Thread {thread_id} not found")
                return False
            
            # Add the message using the existing method
            message_data["thread_id"] = thread_id
            return self.add_new_message(message_data)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add message to thread: {e}")
            return False
        
        
    def update_graph(self, path_or_list: List[Path] = None) -> bool:
        """Update existing graph with additional processed files."""
        files = self._discover_files(path_or_list)
        if not files:
            return False
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.load_graph()
        self.build_graph(files)
        return True

    def save_graph(self, pickle_path: Path = None, save_neo4j: bool = False) -> Optional[Path]:
        """Save graph to disk (NetworkX and optionally Neo4j). Returns pickle path on success."""
        try:
            self.logger.info("💾 Starting MailHierarchyGraph save process...")
            if pickle_path is None:
                pickle_path = self.graph_full_path
                
            if pickle_path is None:
                self.logger.error("pickle_path must be provided for save")
                return None

            # Ensure parent directory exists
            Path(pickle_path).parent.mkdir(parents=True, exist_ok=True)

            # Step 1: Save NetworkX graph
            self.logger.info("📦 Saving NetworkX graph to pickle...")
            try:
                if pickle_path is not None:
                    self.logger.info("📄 Saving NetworkX graph to pickle...")
                    graph_data = {
                    'graph': self.graph,
                    'entity_embeddings': self.node_embeddings,
                    'node_summaries': self.node_summaries,
                    'vector_index': self.vector_index
                }
                
                # Save to file
                with open(pickle_path, 'wb') as f:
                    pickle.dump(graph_data, f)
                
                self.logger.info(f"✅ Saved NetworkX graph to {pickle_path}")
            except Exception as exc:
                self.logger.warning(f"Failed to save pickle graph: {exc}")
            
            
            # Save to Neo4j if requested
            if save_neo4j:
                self.logger.info("🗄️ Saving to Neo4j database...")
                self._save_to_neo4j()

            self.logger.info("🎉 MailHierarchyGraph save completed successfully!")
            self.logger.info(f"📊 Final save statistics:")
            self.logger.info(f"  - NetworkX graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            self.logger.info(f"  - Embeddings: {len(self.node_embeddings)}")
            self.logger.info(f"  - Summaries: {len(self.node_summaries)}")
            self.logger.info(f"  - Neo4j: {'✅ Saved' if save_neo4j else '⏭️ Skipped'}")
            
            return Path(pickle_path)
        except Exception as e:
            self.logger.error(f"Error saving mail hierarchy graph: {e}")
            return None

    def load_graph(self, pickle_path: Path = None) -> bool:
        """Load graph from pickle file with optional embeddings and summaries."""
        try:
            self.logger.info("📁 Starting MailHierarchyGraph load process...")
            if pickle_path is None:
                pickle_path = self.graph_full_path
                
            if not Path(pickle_path).exists():
                pickle_path = self.file_path
            
            if pickle_path.is_dir():

                pickle_files = list(pickle_path.rglob(f"sub_{self.graph_filename}"))
                if not pickle_files:
                    self.logger.warning(f"Graph file not found: {pickle_path}")
                    return False
                self.vector_index = None
                self.index_to_node = []
                for pickle_file in pickle_files:
                    with open(pickle_file, 'rb') as f:
                        graph_data = pickle.load(f)
                    self.graph = nx.compose(self.graph, graph_data['graph'])
                    self.node_embeddings = {**self.node_embeddings, **graph_data['entity_embeddings']}
                    self.node_summaries = {**self.node_summaries, **graph_data.get('node_summaries', self.node_summaries)}
                    
                    if self.vector_index is None:
                        self.vector_index = graph_data.get('vector_index', [])
                    else:
                        self.vector_index.merge_from(graph_data.get('vector_index'))
                    
                    self.index_to_node.extend(list(graph_data['graph'].nodes()))
            else:
                
                with open(pickle_path, 'rb') as f:
                    graph_data = pickle.load(f)
            
                # Restore graph data
                self.graph = graph_data['graph']
                self.node_embeddings = graph_data['entity_embeddings']
                self.node_summaries = graph_data.get('node_summaries', self.node_summaries)
                if self.vector_index is None:
                    self.vector_index = graph_data.get('vector_index', [])
                else:
                    self.vector_index.merge_from(graph_data.get('vector_index', []), shift_ids=True)
                self.index_to_node = list(self.graph.nodes())
            
            
            self.logger.info("🎉 MailHierarchyGraph load completed successfully!")
            self.logger.info(f"📊 Final load statistics:")
            self.logger.info(f"  - NetworkX graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            self.logger.info(f"  - Embeddings: {len(self.node_embeddings)}")
            self.logger.info(f"  - Summaries: {len(self.node_summaries)}")
            self.logger.info(f"  - Vector index: {'✅ Built' if self.vector_index else '❌ Not available'}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading mail hierarchy graph: {e}")
            return False

    def _save_to_neo4j(self):
        """Save graph to Neo4j database."""
        try:
            # Get Neo4j configuration
            neo4j_config = get_config("rag.retrieval.graph.neo4j", {})
            if not neo4j_config:
                self.logger.warning("Neo4j configuration not found, skipping Neo4j save")
                return
            
            # Import Neo4j driver
            try:
                from neo4j import GraphDatabase
            except ImportError:
                self.logger.warning("Neo4j driver not installed, skipping Neo4j save")
                return
            
            # Connect to Neo4j
            driver = GraphDatabase.driver(
                neo4j_config.get("uri", "bolt://localhost:7687"),
                auth=(neo4j_config.get("username", "neo4j"), neo4j_config.get("password", "password"))
            )
            
            with driver.session(database=neo4j_config.get("database", "neo4j")) as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create nodes
                for node_id, attrs in self.graph.nodes(data=True):
                    session.run("""
                        CREATE (n:Message {
                            id: $id,
                            subject: $subject,
                            sender: $sender,
                            date: $date,
                            summary: $summary,
                            thread_name: $thread_name
                        })
                    """, 
                    id=node_id,
                    subject=attrs.get('subject', ''),
                    sender=attrs.get('sender_address', ''),
                    date=attrs.get('date', ''),
                    summary=attrs.get('summary', ''),
                    thread_name=attrs.get('thread_name', '')
                    )
                
                # Create relationships
                for source, target, attrs in self.graph.edges(data=True):
                    relationship_type = attrs.get('relationship', 'RELATED')
                    session.run("""
                        MATCH (a:Message {id: $source})
                        MATCH (b:Message {id: $target})
                        CREATE (a)-[r:%s]->(b)
                    """ % relationship_type, source=source, target=target)
                
                self.logger.info(f"Saved graph to Neo4j: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
            driver.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save to Neo4j: {e}")

    def load_from_neo4j(self) -> bool:
        """Load graph from Neo4j database."""
        try:
            from utils.config import get_config
            
            # Get Neo4j configuration
            neo4j_config = get_config("rag.retrieval.graph.neo4j", {})
            if not neo4j_config:
                self.logger.warning("Neo4j configuration not found")
                return False
            
            # Import Neo4j driver
            try:
                from neo4j import GraphDatabase
            except ImportError:
                self.logger.warning("Neo4j driver not installed")
                return False
            
            # Connect to Neo4j
            driver = GraphDatabase.driver(
                neo4j_config.get("uri", "bolt://localhost:7687"),
                auth=(neo4j_config.get("username", "neo4j"), neo4j_config.get("password", "password"))
            )
            
            self.graph = nx.DiGraph()
            
            with driver.session(database=neo4j_config.get("database", "neo4j")) as session:
                # Load nodes
                result = session.run("MATCH (n:Message) RETURN n")
                for record in result:
                    node_data = record["n"]
                    node_id = node_data["id"]
                    attrs = dict(node_data)
                    del attrs["id"]  # Remove id from attributes
                    self.graph.add_node(node_id, **attrs)
                
                # Load relationships
                result = session.run("MATCH (a:Message)-[r]->(b:Message) RETURN a.id, b.id, type(r)")
                for record in result:
                    source = record["a.id"]
                    target = record["b.id"]
                    rel_type = record["type(r)"]
                    self.graph.add_edge(source, target, relationship=rel_type)
                
                self.logger.info(f"Loaded graph from Neo4j: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
            driver.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load from Neo4j: {e}")
            return False

    def search(self, query: str, max_results: int = 10, 
               filters: Dict[str, Any] = None, include_context: bool = False) -> List[RetrievalResult]:
        """
        Powerful search method for the mail hierarchy graph.
        
        Args:
            query: Search query string
            search_type: Type of search ("semantic", "keyword", "graph", "hybrid")
            max_results: Maximum number of results to return
            filters: Optional filters (sender, thread, date_range, etc.)
            include_context: Whether to include parent/child context
            
        Returns:
            List of search results with relevance scores and metadata
            List of RetrievalResult objects
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            self.logger.warning("Graph is empty, cannot perform search")
            return []
        
        self.logger.info(f"Performing hybrid search for: '{query}'")
        
        # Apply filters first
        filtered_nodes = self._apply_filters(filters) if filters else list(self.graph.nodes())
        
        search_results = self._hybrid_search(query, filtered_nodes, max_results)
        
        # Add context if requested
        if include_context:
            search_results = self._add_context_to_results(search_results)
        
        # Sort by relevance score
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(f"Search completed: {len(search_results)} results found")
        return search_results

    def _apply_filters(self, filters: Dict[str, Any]) -> List[str]:
        """Apply filters to nodes."""
        filtered_nodes = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            
            # Sender filter
            if 'sender' in filters and filters['sender']:
                if not node_data.get('sender') or filters['sender'].lower() not in node_data.get('sender', '').lower():
                    continue
            
            # Thread filter
            if 'thread' in filters and filters['thread']:
                if not node_data.get('thread') or filters['thread'].lower() not in node_data.get('thread', '').lower():
                    continue
            
            # Subject filter
            if 'subject' in filters and filters['subject']:
                if not node_data.get('subject') or filters['subject'].lower() not in node_data.get('subject', '').lower():
                    continue
            
            # Date range filter (if date metadata exists)
            if 'date_range' in filters and filters['date_range']:
                # This would need date parsing logic based on your data structure
                pass
            
            filtered_nodes.append(node_id)
        
        return filtered_nodes

    def _semantic_search(self, query: str, filtered_nodes: List[str], max_results: int) -> List[RetrievalResult]:
        """Perform fast semantic search using pre-computed embeddings and FAISS index."""
        try:
            # Check if we have pre-computed embeddings and vector index
            if not self.vector_index or not self.node_embeddings:
                self.logger.warning("No pre-computed embeddings available, falling back to keyword search")
                return self._keyword_search(query, filtered_nodes, max_results)
            
            # Get query embedding
            query_embedding = self.embedder.encode(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Normalize query embedding for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search in FAISS index
            scores, indices = self.vector_index.search(query_embedding, max(max_results*100, int(len(filtered_nodes)/2)))  # Get more results for filtering
            threshold = scores[0][0] * 0.8
            results: List[RetrievalResult] = []
            for score, idx in zip(scores[0], indices[0]):
                if score < threshold:
                    break
                if idx == -1:  # Invalid index
                    continue
                    
                node_id = self.index_to_node[idx]
                
                # Apply filters if needed
                if filtered_nodes and node_id not in filtered_nodes:
                    continue
                
                # Get node data
                node_data = self.graph.nodes[node_id]
                
                # Create text representation
                node_text = self._create_node_text(node_data)
                
                # Use pre-computed summary if available
                summary = self.node_summaries.get(node_id, node_data.get('summary', ''))
                
                results.append(RetrievalResult(
                    text = node_text,
                    score = float(score),
                    metadata = node_data,
                    retrieval_method = 'hierarchical_semantic',
                    source_type = "email",
                    source_file = node_data.get('url', ''),
                    node_id = node_id,
                ))
                
            if not results:
                return []
            # Recency reweighting (create gen_score)
            self._apply_recency(results, alpha=0.7, beta=0.3, accumulate=False)
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return self._keyword_search(query, filtered_nodes, max_results)

    def _keyword_search(self, query: str, filtered_nodes: List[str], max_results: int) -> List[RetrievalResult]:
        """Perform keyword-based search."""
        query_terms = query.lower().split()
        results: List[RetrievalResult] = []
        
        for node_id in filtered_nodes:
            node_data = self.graph.nodes[node_id]
            
            # Create searchable text
            searchable_text = self._create_node_text(node_data)
            
            # Calculate keyword matches
            matches = sum(1 for term in query_terms if term in searchable_text.lower())
            if matches == 0:
                continue
            
            # Calculate relevance score
            relevance_score = matches / len(query_terms)
            
            # Boost score for exact phrase matches
            if query.lower() in searchable_text.lower():
                relevance_score += 0.5
            
            results.append(RetrievalResult(
                text = searchable_text,
                score = float(relevance_score),
                metadata = node_data,
                retrieval_method = 'hierarchical_keyword',
                source_type = "email",
                source_file = node_data.get('url', ''),
                node_id = node_id,
            ))
        
        
        # filtered.sort(key=lambda x: x.score, reverse=True)
        return self._finalize_results(results, threshold_ratio=0.8, recency=True, accumulate=False, prev_scale=1.0, add_scale=0.3)

    def _graph_search(self, query: str, filtered_nodes: List[str], max_results: int) -> List[RetrievalResult]:
        """Perform graph-based search using embedding similarity over a node and its neighbors."""
        try:
            # Build query embedding
            query_embedding = self.embedder.encode(query)
            if query_embedding is None:
                return []
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            else:
                query_embedding = query_embedding.astype('float32')
            # Normalize for cosine similarity
            q_norm = np.linalg.norm(query_embedding)
            if q_norm == 0:
                return []
        except Exception as e:
            self.logger.warning(f"Failed to build query embedding for graph search: {e}")
            return []
        results: List[RetrievalResult] = []
        
        for node_id in filtered_nodes:
            node_data = self.graph.nodes[node_id]
            
            # Calculate graph-based relevance using embedding similarity synthesis
            graph_score = self._calculate_graph_relevance(node_id, query_embedding)
            
            if graph_score > 0:
                results.append(RetrievalResult(
                    text = self._create_node_text(node_data),
                    score = graph_score,
                    retrieval_method = 'hierarchical_graph',
                    metadata = node_data,
                    source_type = "email",
                    source_file = node_data.get('url', ''),
                    node_id = node_id,
                ))
        
        # Stage 1: thresholding by top raw score
        if not results:
            return []
        
        return self._finalize_results(results, threshold_ratio=0.7, recency=True, accumulate=False, prev_scale=1.0, add_scale=0.3)

    def _hybrid_search(self, query: str, filtered_nodes: List[str], max_results: int) -> List[RetrievalResult]:
        """Perform hybrid search combining multiple methods."""
        # Get results from different search methods
        semantic_results = self._semantic_search(query, filtered_nodes, max_results)
        filtered_nodes = [r.node_id for r in semantic_results]
        keyword_results = self._keyword_search(query, filtered_nodes, max_results)
        filtered_nodes = [r.node_id for r in keyword_results]
        graph_results = self._graph_search(query, filtered_nodes, max_results)
        filtered_nodes = [r.node_id for r in graph_results]
        
        graph_results.sort(key=lambda x: x.gen_score, reverse=True)
        
        return graph_results[:max_results]

    def _compute_recency_weight(self, node_data: Dict[str, Any]) -> float:
        """Compute a recency weight in (0, 1], favoring recent dates.
        Uses exponential decay with configurable half-life (default 180 days).
        Returns 1.0 if no date available or parsing fails.
        """
        try:
            date_str = node_data.get('date') or node_data.get('date_active') or ''
            if not date_str:
                return 1.0
            dt = None
            s = date_str.strip()
            # Try ISO formats
            try:
                if s.endswith('Z'):
                    s = s.replace('Z', '+00:00')
                dt = datetime.fromisoformat(s)
            except Exception:
                dt = None
            # Try common email/date formats
            if dt is None:
                fmts = [
                    '%a, %d %b %Y %H:%M:%S %z',  # RFC 2822
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                ]
                for fmt in fmts:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        break
                    except Exception:
                        continue
            if dt is None:
                return 1.0
            # Normalize to UTC if naive
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
            half_life = float(get_config('rag.retrieval.mail.recency_half_life_days', 1800))
            if half_life <= 0:
                half_life = 180.0
            # Exponential decay: weight=exp(-age/half_life)
            weight = math.exp(-age_days / half_life)
            # Clamp to avoid zeroing out older but relevant content
            return max(0.1, min(1.0, weight))
        except Exception:
            return 1.0

    def _create_node_text(self, node_data: Dict[str, Any]) -> str:
        """Create searchable text from node data."""
        text_parts = []
        
        if node_data.get('subject'):
            text_parts.append(f"subject: {node_data['subject']}")
        
        if node_data.get('summary'):
            text_parts.append(f"summary: {node_data['summary']}")
        
        if node_data.get('sender_address'):
            text_parts.append(f"from: {node_data['sender_address']}")
        
        return '\n'.join(text_parts)

    def _calculate_graph_relevance(self, node_id: str, query_embedding: np.ndarray) -> float:
        """Calculate graph-based relevance by synthesizing cosine similarity over a node and its neighbors.

        Strategy:
        - Compute cosine similarity between query embedding and the current node's embedding.
        - Compute cosine similarity for all immediate neighbors (predecessors and successors).
        - Aggregate as a weighted combination: 0.6 * sim(node) + 0.4 * mean(sim(neighbors)).
          If no neighbors, return node similarity. If no embedding, return 0.
        """
        try:
            # Normalize inputs
            q_vec = self._normalize_vector(query_embedding)
            if q_vec is None:
                return 0.0
            node_vec = self._get_normalized_node_vector(node_id)
            if node_vec is None:
                return 0.0
            node_sim = float(np.dot(node_vec, q_vec))

            # Neighbor embeddings (both directions)
            neighbors = list(self.graph.predecessors(node_id)) + list(self.graph.successors(node_id))
            sims = []
            for nb in neighbors:
                nb_vec = self._get_normalized_node_vector(nb)
                if nb_vec is None:
                    continue
                sims.append(float(np.dot(nb_vec, q_vec)))

            if sims:
                agg = 0.6 * node_sim + 0.4 * (sum(sims) / len(sims))
            else:
                agg = node_sim
            # Clamp to [0, 1] since cosine could be negative; we prefer non-negative relevance
            return max(0.0, min(1.0, agg))
        except Exception as e:
            self.logger.warning(f"Error calculating graph relevance: {e}")
            return 0.0

    def _get_graph_metrics(self, node_id: str) -> Dict[str, Any]:
        """Get graph metrics for a node."""
        try:
            return {
                'degree': self.graph.degree(node_id),
                'in_degree': self.graph.in_degree(node_id),
                'out_degree': self.graph.out_degree(node_id),
                'betweenness_centrality': nx.betweenness_centrality(self.graph).get(node_id, 0),
                'closeness_centrality': nx.closeness_centrality(self.graph).get(node_id, 0)
            }
        except Exception as e:
            self.logger.warning(f"Error calculating graph metrics: {e}")
            return {}

    def _add_context_to_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add parent/child context to search results."""
        for result in results:
            node_id = result['node_id']
            
            # Get parent context
            parents = list(self.graph.predecessors(node_id))
            parent_context = []
            for parent_id in parents[:2]:  # Limit to 2 parents
                parent_data = self.graph.nodes[parent_id]
                parent_context.append({
                    'node_id': parent_id,
                    'subject': parent_data.get('subject', ''),
                    'sender': parent_data.get('sender', '')
                })
            
            # Get child context
            children = list(self.graph.successors(node_id))
            child_context = []
            for child_id in children[:3]:  # Limit to 3 children
                child_data = self.graph.nodes[child_id]
                child_context.append({
                    'node_id': child_id,
                    'subject': child_data.get('subject', ''),
                    'sender': child_data.get('sender', '')
                })
            
            result['context'] = {
                'parents': parent_context,
                'children': child_context
            }
        
        return results

    def search_by_sender(self, sender_email: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for messages by sender email."""
        results = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('sender') and sender_email.lower() in node_data.get('sender', '').lower():
                results.append({
                    'node_id': node_id,
                    'relevance_score': 1.0,
                    'search_type': 'sender',
                    'content': self._create_node_text(node_data),
                    'metadata': node_data
                })
        
        return results[:max_results]

    def search_by_thread(self, thread_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for messages in a specific thread."""
        results = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('thread') and thread_id in node_data.get('thread', ''):
                results.append({
                    'node_id': node_id,
                    'relevance_score': 1.0,
                    'search_type': 'thread',
                    'content': self._create_node_text(node_data),
                    'metadata': node_data
                })
        
        return results[:max_results]

    def get_thread_conversation(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get the complete conversation for a thread."""
        thread_messages = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('thread') and thread_id in node_data.get('thread', ''):
                # Get message context
                parents = list(self.graph.predecessors(node_id))
                children = list(self.graph.successors(node_id))
                
                thread_messages.append({
                    'node_id': node_id,
                    'content': self._create_node_text(node_data),
                    'metadata': node_data,
                    'parents': parents,
                    'children': children,
                    'position_in_thread': len(parents)  # Approximate position
                })
        
        # Sort by position in thread
        thread_messages.sort(key=lambda x: x['position_in_thread'])
        return thread_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the mail hierarchical graph."""
        try:
            stats = {
                "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
                "graph_edges": self.graph.number_of_edges() if self.graph else 0,
                "node_embeddings_count": len(self.node_embeddings),
                "node_summaries_count": len(self.node_summaries),
                "vector_index_available": self.vector_index is not None,
                "language": getattr(self, 'language', 'unknown'),
                "graph_filename": getattr(self, 'graph_filename', 'unknown'),
                "status": "initialized" if self.graph and self.graph.number_of_nodes() > 0 else "empty"
            }
            
            # Add embedding statistics if available
            if self.node_embeddings:
                stats["embedding_dimension"] = len(next(iter(self.node_embeddings.values()))) if self.node_embeddings else 0
            
            # Add vector index statistics if available
            if self.vector_index:
                stats["vector_index_size"] = self.vector_index.ntotal if hasattr(self.vector_index, 'ntotal') else 0
                stats["vector_index_dimension"] = self.vector_index.d if hasattr(self.vector_index, 'd') else 0
            
            # Add hierarchy statistics
            if self.graph:
                # Count different node types
                node_types = {}
                for node_id in self.graph.nodes():
                    node_data = self.graph.nodes[node_id]
                    node_type = node_data.get('node_type', 'unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                stats["node_types"] = node_types
                
                # Count relationship types
                rel_types = {}
                for source, target, data in self.graph.edges(data=True):
                    rel_type = data.get('relationship', 'unknown')
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                stats["relationship_types"] = rel_types
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting mail hierarchical graph stats: {e}")
            return {
                "graph_nodes": 0,
                "graph_edges": 0,
                "node_embeddings_count": 0,
                "node_summaries_count": 0,
                "vector_index_available": False,
                "status": "error",
                "error": str(e)
            }

def main():
    """Main function for testing hybrid retrieval."""
    # This would be used for testing the hybrid retrieval system
    pass


if __name__ == "__main__":
    main()
