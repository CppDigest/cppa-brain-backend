"""
Abstract interface for vector stores.
Supports FAISS and Chroma backends for document embeddings.
"""

from abc import ABC, abstractmethod
from tkinter import NO
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
from loguru import logger
from utils.config import get_config, get_model_nick_name, RetrievalResult
import faiss
import pickle


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""
    
    def __init__(self, embedding_model, language: str = "en"):
        self.embedding_model = embedding_model
        self.language = language
        self.logger = logger.bind(name=self.__class__.__name__)
        self._initialized = False
    
    
    @abstractmethod
    def build_index(self, chunks: List[Dict[str, Any]]) -> bool:
        """Build vector index from chunks."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
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
        """Get statistics about the vector store."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self._initialized


class FAISSVectorStore(VectorStoreInterface):
    """FAISS-based vector store implementation."""
    
    def __init__(self, embedding_model, language: str = "en"):
        super().__init__(embedding_model, language)
        self.index = None
        self.metadata = {}
        self.dimension = None
    
    def build_index(self, chunks: List[Dict[str, Any]]=None) -> bool:
        """Build FAISS index from chunks."""
        try:
            self.logger.info("🔢 Building FAISS index...")
            
            if not chunks:
                return False
            
            # Get embeddings
            texts = [chunk['chunk_text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            # Set dimension
            self.dimension = embeddings.shape[1]
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata['chunks']
            self.metadata['chunks'] = chunks
            
            self._initialized = True
            self.logger.info(f"✅ FAISS index built with {len(chunks)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build FAISS index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search FAISS index."""
        if not self._initialized or self.index is None:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata['chunks']):
                    results.append(RetrievalResult(
                        text=self.metadata['chunks'][idx]['chunk_text'],
                        score=float(score),
                        metadata=self.metadata['chunks'][idx],
                        retrieval_method='faiss',
                        source_type=self.metadata['chunks'][idx].get('source_file', '').split('.')[-1],
                        source_file=self.metadata['chunks'][idx].get('source_file', '')
                    ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ FAISS search failed: {e}")
            return []
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add new documents to FAISS index."""
        try:
            if not chunks:
                return True
            
            self.logger.info(f"📄 Adding {len(chunks)} documents to FAISS index...")
            
            # Get embeddings
            texts = [chunk['chunk_text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            # Add to index
            self.index.add(embeddings.astype('float32'))
            
            # Update metadata['chunks']
            self.metadata['chunks'].extend(chunks)
            
            self.logger.info(f"✅ Added {len(chunks)} documents to FAISS index")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add documents to FAISS: {e}")
            return False
    
    def update_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Update documents in FAISS index."""
        # FAISS doesn't support direct updates, so we rebuild
        return self.build_index(chunks)
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS index."""
        # FAISS doesn't support deletion, so we rebuild without deleted docs
        remaining_chunks = [chunk for chunk in self.metadata['chunks'] 
                          if chunk.get('id', '') not in document_ids]
        return self.build_index(remaining_chunks)
    
    def save_index(self, path: Optional[Path] = None) -> bool:
        """Save FAISS index to disk."""
        try:
            base = self._resolve_base_path(path)
            base.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(base / "faiss_index.bin"))
            
            # Save metadata
            with open(base / "metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata, f)
            
            self.logger.info(f"✅ FAISS index saved to {base}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save FAISS index: {e}")
            return False
    
    def load_index(self, path: Optional[Path] = None) -> bool:
        """Load FAISS index from disk."""
        try:
            base = self._resolve_base_path(path)
            # Prefer new filenames; fall back to old if present
            index_path = base / "faiss_index.bin"
            if not index_path.exists():
                index_path = base / "faiss.index"
            metadata_path = base / "metadata.pkl"
            legacy_meta_json = base / "metadata.json"
            docs_json = base / "documents.json"
            
            if not index_path.exists():
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata['chunks']: try pkl; else try structured json pairs
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            elif legacy_meta_json.exists() and docs_json.exists():
                import json
                with open(legacy_meta_json, 'r', encoding='utf-8') as f:
                    metas = json.load(f)
                with open(docs_json, 'r', encoding='utf-8') as f:
                    docs = json.load(f)
                # Expect docs as list of texts or dicts; metas align by index
                model_name = metas['model_name']
                self.metadata['model_name'] = model_name
                self.metadata['chunks'] = []
                for i in range(metas['total_chunks']):
                    # text = docs[i]['text'] if isinstance(docs[i], dict) and 'text' in docs[i] else (docs[i] if isinstance(docs[i], str) else "")
                    meta = metas['chunks'][i] if isinstance(metas['chunks'], list) else metas.get(str(i), {})
                    self.metadata['chunks'].append(meta)
            else:
                self.logger.warning("⚠️ No metadata['chunks'] file found alongside faiss index; initializing empty metadata['chunks']")
                self.metadata['chunks'] = []
            
            self._initialized = True
            # set dimension if readable
            try:
                self.dimension = self.index.d
            except Exception:
                pass
            self.logger.info(f"✅ FAISS index loaded from {base}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load FAISS index: {e}")
            return False

    def _resolve_base_path(self, override: Optional[Path]) -> Path:
        """Resolve base directory for FAISS data using config with sensible defaults."""
        if override is not None:
            return override
        embeddings_root = get_config("rag.processed_data.embeddings_data_path", "data/processed/embeddings")
        model_nick = get_model_nick_name(self.embedding_model.model_card_data.base_model)
        language_dir = self.language or get_config("language.default_language", "en")
        # Allow per-model override path in config: rag.embedding.{nick}.persist_directory
        per_model_key = f"rag.embedding.{model_nick}.persist_directory"
        per_model_override = get_config(per_model_key, None)
        base = Path(per_model_override or embeddings_root) / model_nick / language_dir
        return base
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        return {
            "type": "faiss",
            "initialized": self._initialized,
            "dimension": self.dimension,
            "document_count": len(self.metadata.get('chunks',[])),
            "index_size": self.index.ntotal if self.index else 0
        }


class ChromaVectorStore(VectorStoreInterface):
    """Chroma-based vector store implementation."""
    
    def __init__(self, embedding_model, language: str = "en", persist_directory: str = None):
        super().__init__(embedding_model, language)
        self.persist_directory = persist_directory or self._resolve_base_path(None)
        self.collection = None
        self.client = None
    
    def _get_client(self):
        """Get or create Chroma client."""
        if self.client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                self.logger.error("❌ ChromaDB not installed. Install with: pip install chromadb")
                raise
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        return self.client

    def _make_embedding_function(self):
        """Wrap self.embedding_model as a Chroma embedding function if available."""
        model = self.embedding_model
        if model is None:
            return None
        
        class _EmbeddingFunction:
            def __init__(self, m):
                self.m = m
            
            def __call__(self, texts):
                try:
                    vectors = self.m.encode(texts)
                    # Ensure python lists of floats
                    if hasattr(vectors, 'tolist'):
                        return vectors.tolist()
                    return [[float(x) for x in v] for v in vectors]
                except Exception:
                    # Fallback to zeros if embedding computation fails
                    dim = getattr(self.m, 'get_sentence_embedding_dimension', lambda: 768)()
                    return [[0.0] * dim for _ in texts]
        
        return _EmbeddingFunction(model)
    
    def _resolve_base_path(self, override: Optional[Path]=None) -> Path:
        """Resolve base directory for Chroma data using config with sensible defaults."""
        if override is not None:
            return override
        embeddings_root = get_config("rag.database.chroma.persist_directory", "data/processed/chroma_db")
        model_nick = get_model_nick_name(self.embedding_model.model_card_data.base_model)
        language_dir = self.language or get_config("language.default_language", "en")
        base = Path(embeddings_root) / model_nick / language_dir
        return base
    
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> bool:
        """Build Chroma index from chunks."""
        try:
            self.logger.info("🔢 Building Chroma index...")
            
            if not chunks:
                self.logger.warning("⚠️ No chunks provided for Chroma index")
                return False
            
            # Get or create collection; bind embedding function from self.embedding_model when possible
            client = self._get_client()
            collection_name = f"documents_{self.language}"
            ef = self._make_embedding_function()
            try:
                # Preferred API (recent Chroma)
                self.collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"language": self.language},
                    embedding_function=ef
                )
                self.logger.info("📁 Opened Chroma collection with custom embedding function")
            except Exception:
                # Fallback for older Chroma versions
                try:
                    self.collection = client.get_collection(collection_name)
                    self.logger.info("📁 Using existing Chroma collection")
                except Exception:
                    self.collection = client.create_collection(
                        name=collection_name,
                        metadata={"language": self.language}
                    )
                    self.logger.info("📁 Created new Chroma collection")
            
            # Prepare data
            texts = [chunk['chunk_text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            ids = [f"doc_{i}" for i in range(len(chunks))]
            
            # Add documents to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self._initialized = True
            self.logger.info(f"✅ Chroma index built with {len(chunks)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build Chroma index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search Chroma index."""
        if not self._initialized or self.collection is None:
            return []
        
        try:
            # Search collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    # Convert distance to similarity score (Chroma uses cosine distance)
                    score = 1 - distance
                    search_results.append(RetrievalResult(
                        text=doc,
                        score=float(score),
                        metadata=metadata,
                        retrieval_method='chroma',
                        source_type=metadata.get('source_file', '').split('.')[-1],
                        source_file=metadata.get('source_file', '')
                    ))
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"❌ Chroma search failed: {e}")
            return []
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add new documents to Chroma index."""
        try:
            if not chunks or not self.collection:
                return True
            
            self.logger.info(f"📄 Adding {len(chunks)} documents to Chroma index...")
            
            # Prepare data
            texts = [chunk['chunk_text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            ids = [f"doc_{len(self.collection.get()['ids']) + i}" for i in range(len(chunks))]
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"✅ Added {len(chunks)} documents to Chroma index")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add documents to Chroma: {e}")
            return False
    
    def update_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Update documents in Chroma index."""
        try:
            if not chunks or not self.collection:
                return True
            
            self.logger.info(f"🔄 Updating {len(chunks)} documents in Chroma index...")
            
            # Prepare data
            texts = [chunk['chunk_text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            ids = [chunk.get('id', f"doc_{i}") for i, chunk in enumerate(chunks)]
            
            # Update collection
            self.collection.update(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"✅ Updated {len(chunks)} documents in Chroma index")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update documents in Chroma: {e}")
            return False
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Chroma index."""
        try:
            if not document_ids or not self.collection:
                return True
            
            self.logger.info(f"🗑️ Deleting {len(document_ids)} documents from Chroma index...")
            
            # Delete from collection
            self.collection.delete(ids=document_ids)
            
            self.logger.info(f"✅ Deleted {len(document_ids)} documents from Chroma index")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to delete documents from Chroma: {e}")
            return False
    
    def save_index(self, path: Path = None) -> bool:
        """Save Chroma index to disk."""
        if path is None:
            path = self._resolve_base_path(None)
            path = path / "chroma.db"
            
        # Chroma automatically persists to disk
        self.logger.info("✅ Chroma index automatically persisted")
        return True
    
    def load_index(self, path: Path = None) -> bool:
        """Load Chroma index from disk."""
        try:
            client = self._get_client()
            collection_name = f"documents_{self.language}"
            ef = self._make_embedding_function()
            try:
                self.collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"language": self.language},
                    embedding_function=ef
                )
            except Exception:
                self.collection = client.get_collection(collection_name)
            self._initialized = True
            
            self.logger.info(f"✅ Chroma index loaded from {self.persist_directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Chroma index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Chroma index statistics."""
        if not self.collection:
            return {
                "type": "chroma",
                "initialized": False,
                "document_count": 0
            }
        
        try:
            collection_data = self.collection.get()
            return {
                "type": "chroma",
                "initialized": self._initialized,
                "document_count": len(collection_data['ids']) if collection_data['ids'] else 0,
                "collection_name": self.collection.name,
                "persist_directory": self.persist_directory
            }
        except:
            return {
                "type": "chroma",
                "initialized": self._initialized,
                "document_count": 0
            }
