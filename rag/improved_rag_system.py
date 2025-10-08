"""
Improved RAG System Architecture
Clean, modular, and testable design with clear separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
from loguru import logger

from rag.vector_store_interface import VectorStoreInterface
from rag.search_engine_interface import SearchEngineInterface
from rag.mail_hierarchical_rag import MailHierarchicalRAG
from rag.document_graph import DocumentGraphRAG
from rag.reranker import CrossEncoderReranker
from utils.config import get_config, RetrievalResult


class SearchScope(Enum):
    """Search scope options."""
    DOCUMENTS = "documents"
    MESSAGES = "messages"
    BOTH = "both"


class SearchMethod(Enum):
    """Search method options."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class SearchRequest:
    """Search request configuration."""
    query: str
    scope: SearchScope = SearchScope.BOTH
    method: SearchMethod = SearchMethod.HYBRID
    max_results: int = 10
    filters: Optional[Dict[str, Any]] = None
    use_reranker: bool = True


@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    results: List[Dict[str, Any]]
    total_found: int
    search_time: float
    method_used: str
    metadata: Dict[str, Any]


class ComponentManager(ABC):
    """Abstract base class for component management."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the component."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if component is ready."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        pass


class VectorStoreManager(ComponentManager):
    """Manages vector store components."""
    
    def __init__(self, embedding_model, language: str = "en", store_type: str = "chroma"):
        self.embedding_model = embedding_model
        self.language = language
        self.store_type = store_type
        self.store: Optional[VectorStoreInterface] = None
        self.logger = logger.bind(name="VectorStoreManager")
    
    def initialize(self) -> bool:
        """Initialize vector store."""
        try:
            from rag.vector_store_interface import FAISSVectorStore, ChromaVectorStore
            
            if self.store_type == "faiss":
                self.store = FAISSVectorStore(self.embedding_model, self.language)
            elif self.store_type == "chroma":
                persist_dir = get_config("rag.database.chroma.persist_directory", f"./chroma_db_{self.language}")
                self.store = ChromaVectorStore(self.embedding_model, self.language, persist_dir)
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
            
            self.logger.info(f"✅ Vector store manager initialized: {self.store_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vector store: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if vector store is ready."""
        return self.store is not None and self.store.is_initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if self.store:
            return self.store.get_stats()
        return {"status": "not_initialized"}


class SearchEngineManager(ComponentManager):
    """Manages search engine components."""
    
    def __init__(self, language: str = "en", engine_type: str = "elasticsearch"):
        self.language = language
        self.engine_type = engine_type
        self.engine: Optional[SearchEngineInterface] = None
        self.logger = logger.bind(name="SearchEngineManager")
    
    def initialize(self) -> bool:
        """Initialize search engine."""
        try:
            from rag.search_engine_interface import BM25SearchEngine, ElasticsearchSearchEngine
            
            if self.engine_type == "bm25":
                self.engine = BM25SearchEngine(self.language)
            elif self.engine_type == "elasticsearch":
                host = get_config("rag.retrieval.search_engine.elasticsearch.host", "localhost")
                port = get_config("rag.retrieval.search_engine.elasticsearch.port", 9200)
                self.engine = ElasticsearchSearchEngine(self.language, host, port)
            else:
                raise ValueError(f"Unsupported search engine type: {self.engine_type}")
            
            self.logger.info(f"✅ Search engine manager initialized: {self.engine_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize search engine: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if search engine is ready."""
        return self.engine is not None and self.engine.is_initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        if self.engine:
            return self.engine.get_stats()
        return {"status": "not_initialized"}


class GraphManager(ComponentManager):
    """Manages graph-based components."""
    
    def __init__(self, embedding_model, language: str = "en", summarizer=None):
        self.embedding_model = embedding_model
        self.language = language
        self.summarizer = summarizer
        self.document_graph: Optional[DocumentGraphRAG] = None
        self.mail_hierarchy: Optional[MailHierarchicalRAG] = None
        self.logger = logger.bind(name="GraphManager")
    
    def initialize(self) -> bool:
        """Initialize graph components."""
        try:
            # Initialize document graph
            self.document_graph = DocumentGraphRAG(
                language=self.language,
                embedder=self.embedding_model
            )
            
            # Initialize mail hierarchy
            self.mail_hierarchy = MailHierarchicalRAG(
                language=self.language,
                embedder=self.embedding_model,  
                summarizer=self.summarizer
            )
            
            self.logger.info("✅ Graph manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize graph manager: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if graph components are ready."""
        return (self.document_graph is not None and 
                self.mail_hierarchy is not None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "document_graph": self.document_graph.get_stats() if self.document_graph else {},
            "mail_hierarchy": self.mail_hierarchy.get_stats() if self.mail_hierarchy else {}
        }


class SearchOrchestrator:
    """Orchestrates search across different components."""
    
    def __init__(self, vector_manager: VectorStoreManager, 
                 search_manager: SearchEngineManager,
                 graph_manager: GraphManager,
                 reranker: Optional[CrossEncoderReranker] = None):
        self.vector_manager = vector_manager
        self.search_manager = search_manager
        self.graph_manager = graph_manager
        self.reranker = reranker
        self.logger = logger.bind(name="SearchOrchestrator")
    
    def search(self, request: SearchRequest) -> SearchResponse:
        """Perform search based on request."""
        import time
        start_time = time.time()
        
        try:
            results = []
            
            # Semantic search
            if request.method in [SearchMethod.SEMANTIC, SearchMethod.HYBRID]:
                if self.vector_manager.is_ready():
                    semantic_results = self.vector_manager.store.search(request.query, request.max_results)
                    results.extend(semantic_results)
            
            # Keyword search
            if request.method in [SearchMethod.KEYWORD, SearchMethod.HYBRID]:
                if self.search_manager.is_ready():
                    keyword_results = self.search_manager.engine.search(request.query, request.max_results)
                    results.extend(keyword_results)
            
            # Graph search
            if request.method in [SearchMethod.GRAPH, SearchMethod.HYBRID]:
                if self.graph_manager.is_ready():
                    # Document graph search
                    doc_results = self.graph_manager.document_graph.search_graph(
                        request.query, self.graph_manager.embedding_model, request.max_results
                    )
                    results.extend(doc_results)

                    # Rerank if enabled
                    if request.use_reranker and self.reranker and results:
                        results = self.reranker.rerank(request.query, results)
                    
                    results = results[:request.max_results]
                    

                    # Mail hierarchy search
                    mail_results = self.graph_manager.mail_hierarchy.search(
                        request.query, request.max_results
                    )
                    mail_results = self.reranker.rerank(request.query, mail_results)
                    
                    results.extend(mail_results[:request.max_results])
            
            search_time = time.time() - start_time
            
            return SearchResponse(
                results=results,
                total_found=len(results),
                search_time=search_time,
                method_used=request.method.value,
                metadata={
                    "scope": request.scope.value,
                    "filters_applied": request.filters is not None
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            return SearchResponse(
                results=[],
                total_found=0,
                search_time=time.time() - start_time,
                method_used=request.method.value,
                metadata={"error": str(e)}
            )


class RAGSystem:
    """
    Improved RAG System with clean architecture.
    
    Features:
    - Clear separation of concerns
    - Modular component management
    - Easy testing and mocking
    - Flexible configuration
    - Better error handling
    """
    
    def __init__(self, language: str = "en", 
                 vector_store_type: str = "chroma",
                 search_engine_type: str = "elasticsearch"):
        self.language = language
        self.logger = logger.bind(name="ImprovedRAGSystem")
        
        # Component managers
        self.vector_manager: Optional[VectorStoreManager] = None
        self.search_manager: Optional[SearchEngineManager] = None
        self.graph_manager: Optional[GraphManager] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.chunk_files: List[str] = []
        self.mail_files: List[str] = []
        
        # Orchestrator
        self.orchestrator: Optional[SearchOrchestrator] = None
        
        # Configuration
        self.vector_store_type = vector_store_type
        self.search_engine_type = search_engine_type
        
        self._initialized = False
    
    def _load_chunks_with_progress(self, chunkfiles_or_path: Path = None) -> List[Dict[str, Any]]:
        """Load chunks with progress tracking."""
        chunks = []
        if chunkfiles_or_path is None:
            chunkfiles_or_path = Path(get_config("data.processed_data.chunked_data_path", "data/processed/chunked"))
            chunkfiles_or_path = chunkfiles_or_path / self.language
            chunk_files = list(chunkfiles_or_path.rglob("*_semantic_chunks.json"))
        elif isinstance(chunkfiles_or_path, list):
            chunk_files = chunkfiles_or_path
        else:
            chunk_files = list(chunkfiles_or_path.rglob("*_semantic_chunks.json"))
        
        self.logger.info(f"📁 Found {len(chunk_files)} chunk files")
        
        for i, chunk_file in enumerate(chunk_files, 1):
            try:
                if i % 50 == 0:
                    progress = (i / len(chunk_files)) * 100
                    self.logger.info(f"  📄 Loading {i}/{len(chunk_files)}: {chunk_file.name} ({progress:.1f}%)")
                
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    file_chunks = json.load(f)
                    if isinstance(file_chunks, list):
                        chunks.extend(file_chunks)
                        
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {chunk_file}: {e}")
        
        return chunks
    
    def initialize(self, embedding_model, summarizer=None, reranker_model=None) -> bool:
        """Initialize the RAG system with models."""
        try:
            self.logger.info("🚀 Initializing Improved RAG System...")
            
            # Initialize vector store manager
            self.vector_manager = VectorStoreManager(
                embedding_model, self.language, self.vector_store_type
            )
            if not self.vector_manager.initialize():
                self.logger.error("❌ Failed to initialize vector store manager")
                return False
            
            # Initialize search engine manager
            self.search_manager = SearchEngineManager(
                self.language, self.search_engine_type
            )
            if not self.search_manager.initialize():
                self.logger.error("❌ Failed to initialize search engine manager")
                return False
            
            # Initialize graph manager
            self.graph_manager = GraphManager(embedding_model, self.language, summarizer)
            if not self.graph_manager.initialize():
                self.logger.error("❌ Failed to initialize graph manager")
                return False
            
            # Initialize reranker
            if reranker_model:
                self.reranker = reranker_model
            
            # Create orchestrator
            self.orchestrator = SearchOrchestrator(
                self.vector_manager,
                self.search_manager,
                self.graph_manager,
                self.reranker
            )
            
            self._initialized = True
            self.logger.info("✅ Improved RAG System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RAG system: {e}")
            return False
    
    def search(self, query: str, scope: SearchScope = SearchScope.BOTH,
               method: SearchMethod = SearchMethod.HYBRID,
               max_results: int = 10, filters: Optional[Dict[str, Any]] = None,
               use_reranker: bool = True) -> SearchResponse:
        """Perform search with the improved system."""
        if not self._initialized:
            self.logger.warning("⚠️ RAG system not initialized")
            return SearchResponse(
                results=[], total_found=0, search_time=0.0,
                method_used=method.value, metadata={"error": "not_initialized"}
            )
        
        request = SearchRequest(
            query=query,
            scope=scope,
            method=method,
            max_results=max_results,
            filters=filters,
            use_reranker=use_reranker
        )
        
        return self.orchestrator.search(request)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "language": self.language,
            "vector_store": self.vector_manager.get_stats() if self.vector_manager else {},
            "search_engine": self.search_manager.get_stats() if self.search_manager else {},
            "graph_components": self.graph_manager.get_stats() if self.graph_manager else {},
            "reranker_available": self.reranker is not None
        }
    
    def load_data(self, chunk_files: List[str] = None, mail_files: List[str] = None) -> bool:
        """Load initial data into the system.
        Strategy:
        1) Try to load persisted indices/graphs first (vector, search engine, document graph, mail hierarchy).
        2) If any required component is missing or failed to load, build from provided files or auto-discovered defaults.
        """
        if not self._initialized:
            self.logger.error("❌ RAG system not initialized")
            return False
        
        try:
            self.logger.info("🔄 Loading data into RAG system (load-first strategy)...")
            
            # 1) Try to load persisted components
            load_status = self._try_load_persisted_components()
            
            # If everything loaded successfully, we're done
            if not any(load_status.values()):
                self.logger.info("✅ All persisted components loaded; skipping build from source files")
                return True

            # 2) Build missing components from source files
            if not self._build_missing_components(load_status, chunk_files, mail_files):
                return False

            # 3) Persist newly built components
            self._persist_components()

            self.logger.info("✅ Data loading completed successfully (load/build)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load data: {e}")
            return False
    
    def _try_load_persisted_components(self) -> Dict[str, bool]:
        """Try to load persisted components. Returns dict indicating which components need building."""
        return {
            'need_build_vector': self._try_load_vector_index(),
            'need_build_search': self._try_load_search_index(),
            'need_build_doc_graph': self._try_load_document_graph(),
            'need_build_mail_graph': self._try_load_mail_graph(),
        }

    def _try_load_vector_index(self) -> bool:
        """Return True if vector index needs building."""
        store = getattr(self.vector_manager, 'store', None) if self.vector_manager else None
        if self.vector_manager and hasattr(store, 'load_index'):
            try:
                if store.load_index():
                    self.logger.info("✅ Loaded persisted vector index")
                    return False
                
                self.logger.warning("⚠️ Vector index load returned False, building index")
                if len(self.chunk_files) == 0:
                    self.chunk_files = self._load_chunks_with_progress()
                store.build_index(self.chunk_files)
                self.logger.info("✅ Built vector index")
                return False
            except Exception as e:
                self.logger.warning(f"⚠️ Vector index load failed, will build: {e}")
        return True

    def _try_load_search_index(self) -> bool:
        """Return True if search index needs building."""
        engine = getattr(self.search_manager, 'engine', None) if self.search_manager else None
        if self.search_manager and hasattr(engine, 'load_index'):
            try:
                if engine.load_index():
                    self.logger.info("✅ Loaded persisted search index")
                    return False
                self.logger.warning("⚠️ Search index load returned False, building index")
                if len(self.chunk_files) == 0:
                    self.chunk_files = self._load_chunks_with_progress()
                engine.build_index(self.chunk_files)
                self.logger.info("✅ Built search index")
                return False
            except Exception as e:
                self.logger.warning(f"⚠️ Search index load failed, will build: {e}")
        return True

    def _try_load_document_graph(self) -> bool:
        """Return True if document graph needs building."""
        doc_graph = getattr(self.graph_manager, 'document_graph', None) if self.graph_manager else None
        if doc_graph and hasattr(doc_graph, 'load_graph'):
            try:
                if doc_graph.load_graph():
                    self.logger.info("✅ Loaded persisted document graph")
                    return False
                self.logger.warning("⚠️ Document graph load returned False, building graph")
                if len(self.chunk_files) == 0:
                    self.chunk_files = self._load_chunks_with_progress()
                doc_graph.build_graph(self.chunk_files)
                self.logger.info("✅ Built document graph")
                return False
            except Exception as e:
                self.logger.warning(f"⚠️ Document graph load failed, will build: {e}")
        return True

    def _try_load_mail_graph(self) -> bool:
        """Return True if mail hierarchy needs building."""
        mail_graph = getattr(self.graph_manager, 'mail_hierarchy', None) if self.graph_manager else None
        if mail_graph and hasattr(mail_graph, 'load_graph'):
            try:
                if mail_graph.load_graph():
                    self.logger.info("✅ Loaded persisted mail hierarchy graph")
                    return False
                self.logger.warning("⚠️ Mail hierarchy load returned False, building graph")
                if len(self.mail_files) == 0:
                    self.mail_files = self._discover_mail_files()
                mail_graph.build_graph(self.mail_files)
                self.logger.info("✅ Built mail hierarchy graph")
                return False
            except Exception as e:
                self.logger.warning(f"⚠️ Mail hierarchy load failed, will build: {e}")
        return True
    
    def _build_missing_components(self, load_status: Dict[str, bool], chunk_files: List[str], mail_files: List[str]) -> bool:
        """Build missing components from source files."""
        if not self._build_document_side_if_needed(load_status, chunk_files):
            return False
        if not self._build_mail_side_if_needed(load_status, mail_files):
            return False
        return True

    def _build_document_side_if_needed(self, load_status: Dict[str, bool], chunk_files: List[str]) -> bool:
        """Build vector/search/doc-graph if needed."""
        if not any([load_status['need_build_vector'], load_status['need_build_search'], load_status['need_build_doc_graph']]):
            return True
        files = chunk_files or self._discover_chunk_files()
        if not files:
            self.logger.warning("⚠️ No chunk files found to build indices/graph")
            return False
        return self._load_document_data(files)

    def _build_mail_side_if_needed(self, load_status: Dict[str, bool], mail_files: List[str]) -> bool:
        """Build mail hierarchy if needed."""
        if not load_status['need_build_mail_graph']:
            return True
        files = mail_files or self._discover_mail_files()
        if not files:
            self.logger.warning("⚠️ No mail files found to build mail hierarchy")
            return False
        return self._load_mail_data(files)
    
    def _persist_components(self):
        """Persist newly built components if supported."""
        self._persist_vector_index()
        self._persist_search_index()
        self._persist_document_graph()
        self._persist_mail_graph()

    def _persist_vector_index(self):
        store = getattr(self.vector_manager, 'store', None) if self.vector_manager else None
        if self.vector_manager and self.vector_manager.is_ready() and hasattr(store, 'save_index'):
            try:
                store.save_index()
                self.logger.debug("💾 Saved vector index")
            except Exception as e:
                self.logger.debug(f"Could not save vector index: {e}")

    def _persist_search_index(self):
        engine = getattr(self.search_manager, 'engine', None) if self.search_manager else None
        if self.search_manager and self.search_manager.is_ready() and hasattr(engine, 'save_index'):
            try:
                engine.save_index()
                self.logger.debug("💾 Saved search index")
            except Exception as e:
                self.logger.debug(f"Could not save search index: {e}")

    def _persist_document_graph(self):
        doc_graph = getattr(self.graph_manager, 'document_graph', None) if self.graph_manager else None
        if doc_graph and hasattr(doc_graph, 'save_graph'):
            try:
                doc_graph.save_graph()
                self.logger.debug("💾 Saved document graph")
            except Exception as e:
                self.logger.debug(f"Could not save document graph: {e}")

    def _persist_mail_graph(self):
        mail_graph = getattr(self.graph_manager, 'mail_hierarchy', None) if self.graph_manager else None
        if mail_graph and hasattr(mail_graph, 'save_graph'):
            try:
                mail_graph.save_graph()
                self.logger.debug("💾 Saved mail hierarchy")
            except Exception as e:
                self.logger.debug(f"Could not save mail hierarchy: {e}")

    def _discover_chunk_files(self) -> List[str]:
        """Discover chunk files from default configured directory for the current language."""
        base = Path(get_config("rag.processed_data.chunked_data_path", "data/processed/chunked")) / self.language
        if not base.exists():
            return []
        patterns = ["*_semantic_chunks.json", "*.chunks.json", "*.json"]
        files: List[str] = []
        try:
            for pattern in patterns:
                files.extend([str(p) for p in base.rglob(pattern)])
            # Prefer semantic chunk files first
            files = sorted(set(files))
        except Exception:
            return files
        return files

    def _discover_mail_files(self) -> List[str]:
        """Discover mail thread files from default configured directory for the current language."""
        base = Path(get_config("rag.processed_data.message_by_thread_path", "data/processed/message_by_thread")) / self.language
        if not base.exists():
            return []
        pattern = "*_thread_*.json"
        files: List[str] = []
        try:
            files: List[str] = [str(p) for p in base.rglob(pattern)]
        except Exception:
            return files
        return files
    
    def _load_document_data(self, chunk_files: List[str]) -> bool:
        """Load document chunks into vector store and search engine."""
        try:
            self.logger.info("📄 Loading document chunks...")
            
            # Load chunks from files
            all_chunks = []
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                        if isinstance(chunks, list):
                            all_chunks.extend(chunks)
                        else:
                            all_chunks.append(chunks)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {chunk_file}: {e}")
                    continue
            
            if not all_chunks:
                self.logger.warning("⚠️ No chunks loaded")
                return True
            
            # Update vector store
            if self.vector_manager and self.vector_manager.is_ready():
                self.logger.info("🔢 Building vector index...")
                if not self.vector_manager.store.build_index(all_chunks):
                    self.logger.error("❌ Failed to build vector index")
                    return False
            
            # Update search engine
            if self.search_manager and self.search_manager.is_ready():
                self.logger.info("🔍 Building search index...")
                if not self.search_manager.engine.build_index(all_chunks):
                    self.logger.error("❌ Failed to build search index")
                    return False
            
            # Update document graph
            if self.graph_manager and self.graph_manager.document_graph:
                self.logger.info("🕸️ Building document graph...")
                if not self.graph_manager.document_graph.build_graph(all_chunks):
                    self.graph_manager.document_graph.save_graph()
                    self.logger.warning("⚠️ Document graph build failed (may need embedder)")
            
            self.logger.info(f"✅ Loaded {len(all_chunks)} document chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load document data: {e}")
            return False
    
    def _load_mail_data(self, mail_files: List[str]) -> bool:
        """Load mail data into mail hierarchy graph."""
        try:
            self.logger.info("📧 Loading mail data...")
            
            if not self.graph_manager or not self.graph_manager.mail_hierarchy:
                self.logger.warning("⚠️ Mail hierarchy not available")
                return True
            
            # Load mail files
            all_mail_data = []
            for mail_file in mail_files:
                try:
                    with open(mail_file, 'r', encoding='utf-8') as f:
                        mail_data = json.load(f)
                        if isinstance(mail_data, list):
                            all_mail_data.extend(mail_data)
                        else:
                            all_mail_data.append(mail_data)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {mail_file}: {e}")
                    continue
            
            if not all_mail_data:
                self.logger.warning("⚠️ No mail data loaded")
                return True
            
            # Build mail hierarchy graph
            if not self.graph_manager.mail_hierarchy.build_graph(all_mail_data):
                self.logger.error("❌ Failed to build mail hierarchy")
                return False
            
            self.logger.info(f"✅ Loaded {len(all_mail_data)} mail threads")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load mail data: {e}")
            return False
    
    def update_data(self, chunk_files: List[str] = None, mail_files: List[str] = None) -> bool:
        """Update system with new data."""
        if not self._initialized:
            return False
        
        try:
            self.logger.info("🔄 Updating RAG system with new data...")
            
            # Update document data
            if chunk_files:
                if not self._update_document_data(chunk_files):
                    return False
            
            # Update mail data
            if mail_files:
                if not self._update_mail_data(mail_files):
                    return False
            
            self.logger.info("✅ RAG system updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update RAG system: {e}")
            return False
    
    def _update_document_data(self, chunk_files: List[str]) -> bool:
        """Update document data with new chunks."""
        try:
            self.logger.info("📄 Updating document data...")
            
            # Load new chunks
            new_chunks = []
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                        if isinstance(chunks, list):
                            new_chunks.extend(chunks)
                        else:
                            new_chunks.append(chunks)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {chunk_file}: {e}")
                    continue
            
            if not new_chunks:
                self.logger.warning("⚠️ No new chunks to update")
                return True
            
            # Update vector store
            if self.vector_manager and self.vector_manager.is_ready():
                self.logger.info("🔢 Updating vector index...")
                if not self.vector_manager.store.add_documents(new_chunks):
                    self.logger.error("❌ Failed to update vector index")
                    return False
            
            # Update search engine
            if self.search_manager and self.search_manager.is_ready():
                self.logger.info("🔍 Updating search index...")
                if not self.search_manager.engine.add_documents(new_chunks):
                    self.logger.error("❌ Failed to update search index")
                    return False
            
            # Update document graph
            if self.graph_manager and self.graph_manager.document_graph:
                self.logger.info("🕸️ Updating document graph...")
                if not self.graph_manager.document_graph.update_graph(new_chunks):
                    self.logger.warning("⚠️ Document graph update failed (may need embedder)")
            
            self.logger.info(f"✅ Updated with {len(new_chunks)} new document chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update document data: {e}")
            return False
    
    def _update_mail_data(self, mail_files: List[str]) -> bool:
        """Update mail data with new mail threads."""
        try:
            self.logger.info("📧 Updating mail data...")
            
            if not self.graph_manager or not self.graph_manager.mail_hierarchy:
                self.logger.warning("⚠️ Mail hierarchy not available")
                return True
            
            # Load new mail data
            new_mail_data = []
            for mail_file in mail_files:
                try:
                    with open(mail_file, 'r', encoding='utf-8') as f:
                        mail_data = json.load(f)
                        if isinstance(mail_data, list):
                            new_mail_data.extend(mail_data)
                        else:
                            new_mail_data.append(mail_data)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {mail_file}: {e}")
                    continue
            
            if not new_mail_data:
                self.logger.warning("⚠️ No new mail data to update")
                return True
            
            # Update mail hierarchy graph
            if not self.graph_manager.mail_hierarchy.update_graph(new_mail_data):
                self.logger.error("❌ Failed to update mail hierarchy")
                return False
            
            self.logger.info(f"✅ Updated with {len(new_mail_data)} new mail threads")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update mail data: {e}")
            return False
