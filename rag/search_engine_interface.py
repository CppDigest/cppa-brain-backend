"""
Abstract interface for search engines.
Supports BM25 and Elasticsearch backends for keyword search.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from utils.config import get_config, RetrievalResult
import json


class SearchEngineInterface(ABC):
    """Abstract interface for search engines."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.logger = logger.bind(name=self.__class__.__name__)
        self._initialized = False
    
    
    
    @abstractmethod
    def build_index(self, chunks: List[Dict[str, Any]]) -> bool:
        """Build search index from chunks."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        pass
    
    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add new documents to existing index."""
        pass
    
    @abstractmethod
    def update_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Update existing documents in index."""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from index."""
        pass
    
    @abstractmethod
    def save_index(self, path: Optional[Path] = None) -> bool:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load_index(self, path: Optional[Path] = None) -> bool:
        """Load index from disk."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search engine."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if search engine is initialized."""
        return self._initialized


class BM25SearchEngine(SearchEngineInterface):
    """BM25-based search engine implementation."""
    
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.bm25 = None
        self.corpus = []
        self.metadata = []
    
    def build_index(self, chunks: List[Dict[str, Any]]=None) -> bool:
        """Build BM25 index from chunks."""
        try:
            self.logger.info("🔍 Building BM25 index...")
            
            if not chunks:
                chunks = self._load_chunks_with_progress()
            
            # Import BM25 here to avoid dependency issues
            from rank_bm25 import BM25Okapi
            import re
            
            # Process chunks
            self.corpus = []
            self.metadata = []
            
            for chunk in chunks:
                text = chunk['text']
                # Simple tokenization
                tokens = re.findall(r'\b\w+\b', text.lower())
                self.corpus.append(tokens)
                self.metadata.append(chunk)
            
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
                self._initialized = True
                self.logger.info(f"✅ BM25 index built with {len(self.corpus)} documents")
                self.save_index()
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to build BM25 index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search BM25 index."""
        if not self._initialized or not self.bm25:
            return []
        
        try:
            import re
            import numpy as np
            
            # Tokenize query
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include positive scores
                    chunk = self.metadata[idx]
                    results.append(RetrievalResult(
                        text=chunk['text'],
                        score=float(scores[idx]),
                        metadata=chunk['metadata'],
                        retrieval_method='bm25',
                        source_type=chunk['metadata'].get('source_file', '').split('.')[-1],
                        source_file=chunk['metadata'].get('source_file', '')
                    ))
            return results
            
        except Exception as e:
            self.logger.error(f"❌ BM25 search failed: {e}")
            return []
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add new documents to BM25 index."""
        try:
            if not chunks:
                return True
            
            self.logger.info(f"📄 Adding {len(chunks)} documents to BM25 index...")
            
            # Process new chunks
            new_corpus = []
            new_metadata = []
            
            import re
            for chunk in chunks:
                text = chunk['text']
                tokens = re.findall(r'\b\w+\b', text.lower())
                new_corpus.append(tokens)
                new_metadata.append(chunk)
            
            # Add to existing corpus
            self.corpus.extend(new_corpus)
            self.metadata.extend(new_metadata)
            
            # Rebuild BM25 index
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus)
            
            self.logger.info(f"✅ Added {len(chunks)} documents to BM25 index")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add documents to BM25: {e}")
            return False
    
    def update_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Update documents in BM25 index."""
        # BM25 doesn't support direct updates, so we rebuild
        return self.build_index(chunks)
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from BM25 index."""
        # BM25 doesn't support deletion, so we rebuild without deleted docs
        remaining_chunks = [chunk for chunk in self.metadata 
                          if chunk.get('id', '') not in document_ids]
        return self.build_index(remaining_chunks)
    
    def save_index(self, path: Optional[Path] = None) -> bool:
        """Save BM25 index to disk."""
        try:
            base = self._resolve_bm25_base(path)
            base.mkdir(parents=True, exist_ok=True)
            
            # Save corpus and metadata
            with open(base / "corpus.json", 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False)
            
            with open(base / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False)
            
            self.logger.info(f"✅ BM25 index saved to {base}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save BM25 index: {e}")
            return False
    
    def load_index(self, path: Optional[Path] = None) -> bool:
        """Load BM25 index from disk."""
        try:
            base = self._resolve_bm25_base(path)
            corpus_path = base / "corpus.json"
            metadata_path = base / "metadata.json"
            
            if not corpus_path.exists() or not metadata_path.exists():
                return False
            
            # Load corpus and metadata
            with open(corpus_path, 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Rebuild BM25 index
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus)
            
            self._initialized = True
            self.logger.info(f"✅ BM25 index loaded from {base}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load BM25 index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics."""
        return {
            "type": "bm25",
            "initialized": self._initialized,
            "corpus_size": len(self.corpus) if self.corpus else 0,
            "metadata_size": len(self.metadata) if self.metadata else 0
        }

    def _resolve_bm25_base(self, override: Optional[Path]) -> Path:
        if override is not None:
            return override
        root = get_config("rag.retrieval.search_engine.bm25.persist_directory", "data/processed/bm25_index")
        lang = self.language or get_config("language.default_language", "en")
        return Path(root) / lang


class ElasticsearchSearchEngine(SearchEngineInterface):
    """Elasticsearch-based search engine implementation."""
    
    def __init__(self, language: str = "en", host: str = "localhost", port: int = 9200):
        super().__init__(language)
        self.host = host
        self.port = port
        self.client = None
        base_name = get_config("rag.retrieval.search_engine.elasticsearch.index_name", "documents")
        self.index_name = f"{base_name}_{language}"
    
    def _get_client(self):
        """Get or create Elasticsearch client."""
        if self.client is None:
            try:
                from elasticsearch import Elasticsearch
                self.client = Elasticsearch([{'host': self.host, 'port': self.port}])
                
                # Test connection
                if not self.client.ping():
                    raise ConnectionError("Cannot connect to Elasticsearch")
                
                self.logger.info(f"✅ Connected to Elasticsearch at {self.host}:{self.port}")
            except ImportError:
                self.logger.error("❌ Elasticsearch package not installed. Install with: pip install elasticsearch")
                raise
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
                raise
        
        return self.client
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> bool:
        """Build Elasticsearch index from chunks."""
        try:
            self.logger.info("🔍 Building Elasticsearch index...")
            
            if not chunks:
                self.logger.warning("⚠️ No chunks provided for Elasticsearch index")
                return False
            
            client = self._get_client()
            
            # Create index if it doesn't exist
            if not client.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text", "analyzer": "standard"},
                            "metadata": {"type": "object"},
                            "source_file": {"type": "keyword"},
                            "chunk_id": {"type": "keyword"}
                        }
                    }
                }
                client.indices.create(index=self.index_name, body=mapping)
                self.logger.info(f"📁 Created Elasticsearch index: {self.index_name}")
            
            # Index documents
            for i, chunk in enumerate(chunks):
                doc_id = chunk.get('id', f"doc_{i}")
                document = {
                    "text": chunk['text'],
                    "metadata": chunk['metadata'],
                    "source_file": chunk['metadata'].get('source_file', ''),
                    "chunk_id": doc_id
                }
                
                client.index(
                    index=self.index_name,
                    id=doc_id,
                    body=document
                )
            
            # Refresh index to make documents searchable
            client.indices.refresh(index=self.index_name)
            
            self._initialized = True
            self.logger.info(f"✅ Elasticsearch index built with {len(chunks)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build Elasticsearch index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search Elasticsearch index."""
        if not self._initialized:
            return []
        
        try:
            client = self._get_client()
            
            # Build search query
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "metadata.*"],
                        "type": "best_fields"
                    }
                },
                "size": top_k,
                "_source": ["text", "metadata", "source_file", "chunk_id"]
            }
            
            # Execute search
            response = client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append(RetrievalResult(
                    text=hit['_source']['text'],
                    score=float(hit['_score']),
                    metadata=hit['_source']['metadata'],
                    retrieval_method='elasticsearch',
                    source_type=hit['_source']['metadata'].get('source_file', '').split('.')[-1],
                    source_file=hit['_source']['metadata'].get('source_file', '')
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Elasticsearch search failed: {e}")
            return []
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add new documents to Elasticsearch index."""
        try:
            if not chunks:
                return True
            
            self.logger.info(f"📄 Adding {len(chunks)} documents to Elasticsearch index...")
            
            client = self._get_client()
            
            # Bulk index documents
            actions = []
            for chunk in chunks:
                doc_id = chunk.get('id', f"doc_{len(chunks)}")
                action = {
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": {
                        "text": chunk['text'],
                        "metadata": chunk['metadata'],
                        "source_file": chunk['metadata'].get('source_file', ''),
                        "chunk_id": doc_id
                    }
                }
                actions.append(action)
            
            # Use bulk API for efficiency
            try:
                from elasticsearch.helpers import bulk
                bulk(client, actions)
            except ImportError:
                self.logger.error("❌ Elasticsearch not installed. Install with: pip install elasticsearch")
                raise
            
            # Refresh index
            client.indices.refresh(index=self.index_name)
            
            self.logger.info(f"✅ Added {len(chunks)} documents to Elasticsearch index")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add documents to Elasticsearch: {e}")
            return False
    
    def update_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Update documents in Elasticsearch index."""
        try:
            if not chunks:
                return True
            
            self.logger.info(f"🔄 Updating {len(chunks)} documents in Elasticsearch index...")
            
            client = self._get_client()
            
            # Update documents
            for chunk in chunks:
                doc_id = chunk.get('id', f"doc_{chunk.get('id', 'unknown')}")
                document = {
                    "text": chunk['text'],
                    "metadata": chunk['metadata'],
                    "source_file": chunk['metadata'].get('source_file', ''),
                    "chunk_id": doc_id
                }
                
                client.index(
                    index=self.index_name,
                    id=doc_id,
                    body=document
                )
            
            # Refresh index
            client.indices.refresh(index=self.index_name)
            
            self.logger.info(f"✅ Updated {len(chunks)} documents in Elasticsearch index")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update documents in Elasticsearch: {e}")
            return False
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Elasticsearch index."""
        try:
            if not document_ids:
                return True
            
            self.logger.info(f"🗑️ Deleting {len(document_ids)} documents from Elasticsearch index...")
            
            client = self._get_client()
            
            # Delete documents
            for doc_id in document_ids:
                client.delete(
                    index=self.index_name,
                    id=doc_id,
                    ignore=[404]  # Ignore if document doesn't exist
                )
            
            # Refresh index
            client.indices.refresh(index=self.index_name)
            
            self.logger.info(f"✅ Deleted {len(document_ids)} documents from Elasticsearch index")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to delete documents from Elasticsearch: {e}")
            return False
    
    def save_index(self, path: Path) -> bool:
        """Save Elasticsearch index to disk."""
        # Elasticsearch automatically persists to disk
        self.logger.info("✅ Elasticsearch index automatically persisted")
        return True
    
    def load_index(self, path: Path) -> bool:
        """Load Elasticsearch index from disk."""
        try:
            client = self._get_client()
            
            # Check if index exists
            if client.indices.exists(index=self.index_name):
                self._initialized = True
                self.logger.info(f"✅ Elasticsearch index loaded: {self.index_name}")
                return True
            else:
                self.logger.warning(f"⚠️ Elasticsearch index not found: {self.index_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load Elasticsearch index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Elasticsearch index statistics."""
        try:
            client = self._get_client()
            
            if not client.indices.exists(index=self.index_name):
                return {
                    "type": "elasticsearch",
                    "initialized": False,
                    "document_count": 0
                }
            
            # Get index stats
            stats = client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            
            return {
                "type": "elasticsearch",
                "initialized": self._initialized,
                "document_count": doc_count,
                "index_name": self.index_name,
                "host": self.host,
                "port": self.port
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get Elasticsearch stats: {e}")
            return {
                "type": "elasticsearch",
                "initialized": False,
                "document_count": 0
            }
