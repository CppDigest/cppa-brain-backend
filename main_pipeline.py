"""
Improved Boost Pipeline Architecture
Clean, modular, and testable design with clear separation of concerns.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from loguru import logger
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import get_config, load_config, get_model_nick_name
from data_processor.multiformat_processor import MultiFormatProcessor
from data_processor.semantic_chunker import SemanticChunker
from data_processor.summarize_processor import SummarizePocessor
from data_processor.mail_json_processor import MailJsonProcessor
from rag.improved_rag_system import RAGSystem
from text_generation.text_generation_model import TextGenerationModelFactory
from text_generation.rag_answer_generator import RAGAnswerGenerator
from rag.reranker import CrossEncoderReranker
from rag.simple_rag_evaluator import SimpleRAGEvaluator

class PipelineStage(Enum):
    """Pipeline stage options."""
    SCRAPING = "scraping"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    GRAPH = "graph"
    COMPLETE = "complete"


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    language: str = "en"
    max_files: Optional[int] = None
    max_depth: int = 2
    delay: float = 1.0
    source_url: Optional[str] = None
    output_dir: Optional[str] = None


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


class ModelManager(ComponentManager):
    """Manages all model types with clean separation."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.logger = logger.bind(name="ModelManager")
        
        # Model instances
        self.embedding_model = None
        self.summarizing_model = None
        self.reranker_model = None
        self.text_generation_model = None
        self.rag_answer_generator = None
        
        # Model configuration
        self.models = {
            "embedding": {
                "model_name": get_config(f"rag.embedding.{get_config('rag.embedding.default_embedding_type', 'gemma')}.model_name",
                                       "google/embeddinggemma-300m"),
                "model_type": "embedding",
                "description": "Vector embedding model for semantic search"
            },
            "summarizing": {
                "model_name": get_config("rag.summarizer.model_name", "facebook/bart-large-cnn"),
                "model_type": "summarization", 
                "description": "Summarization model for hierarchical RAG"
            },
            "reranker": {
                "model_name": get_config("rag.reranker.model_name", "Alibaba-NLP/gte-multilingual-reranker-base"),
                "model_type": "reranker",
                "description": "Cross-encoder model for result reranking"
            },
            "text_generation": {
                "model_name": get_config(f"rag.llm.{get_config('rag.llm.default_llm_type', 'ollama')}.model_name", "gemma3:1b"),
                "model_type": "text_generation",
                "model_group": get_config('rag.llm.default_llm_type', 'ollama'),
                "description": "Large Language Model for text generation"
            }
        }
    
    def initialize(self) -> bool:
        """Initialize all models."""
        try:
            self.logger.info("🏭 Initializing all models...")
            
            success = True
            
            # Create embedding model
            if not self._create_embedding_model():
                success = False
            
            # Create text generation model
            if not self._create_text_generation_model():
                success = False
            
            # Create RAG answer generator
            if not self._create_rag_answer_generator():
                success = False
                
            # Create reranker model
            if not self._create_reranker_model():
                success = False

            # Create summarizing model only if enabled
            if get_config("rag.retrieval.hierarchical.use_specialized_summarization", True):
                if not self._create_summarizing_model():
                    success = False
            else:
                self.logger.info("Skipping summarizing model (disabled by config)")
            
            if success:
                self.logger.info("✅ All models initialized successfully")
            else:
                self.logger.warning("⚠️ Some models failed to initialize, but pipeline will continue with fallbacks")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize models: {e}")
            return False
    
    def get_available_embedding_models(self):
        """Get list of available embedding models."""
        embedding_models = []
        for nick_name in get_config("rag.embedding.embedding_types_list", ["gemma", "minilm", "nomic", "jina", "baai"]):
            model_name = get_config(f"rag.embedding.{nick_name}.model_name")
            embedding_models.append(model_name)
        return embedding_models

    def get_embedding_nick_name_from_model_name(self, model_name: str=None):
        """Get the embedding nick name from the model name."""
        if self.embedding_model is None:
            return None
        return get_model_nick_name(self.embedding_model.model_card_data.base_model)

        
        
    def _create_embedding_model(self) -> bool:
        """Create embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.models["embedding"]["model_name"]
            self.embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
            self.logger.info(f"✅ Embedding model created: {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create embedding model: {e}")
            return False
    
    def _create_summarizing_model(self) -> bool:
        """Create summarizing model."""
        try:
            model_name = self.models["summarizing"]["model_name"]
            self.summarizing_model = SummarizePocessor(model_name=model_name)
            self.logger.info(f"✅ Summarizing model created: {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create summarizing model: {e}")
            self.summarizing_model = None
            return False
    
    def _create_reranker_model(self) -> bool:
        """Create reranker model."""
        try:
            model_name = self.models["reranker"]["model_name"]
            self.reranker_model = CrossEncoderReranker(model_name, language=self.language)
            self.logger.info(f"✅ Reranker model created: {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create reranker model: {e}")
            self.reranker_model = None
            return False
    
    def _create_text_generation_model(self) -> bool:
        """Create text generation model."""
        try:
            model_group = self.models["text_generation"]["model_group"]
            model_name = self.models["text_generation"]["model_name"]
            
            self.text_generation_model = TextGenerationModelFactory.create_model(
                model_group=model_group,
                model_name=model_name,
                config=get_config("rag.llm", {})
            )
            self.logger.info(f"✅ Text generation model created: {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create text generation model: {e}")
            self.text_generation_model = None
            return False
    
    def _create_rag_answer_generator(self) -> bool:
        """Create RAG answer generator."""
        try:
            if self.text_generation_model is None:
                self.logger.warning("⚠️ Cannot create RAG answer generator: text generation model not available")
                self.rag_answer_generator = None
                return False
            
            self.rag_answer_generator = RAGAnswerGenerator(self.text_generation_model)
            self.logger.info("✅ RAG answer generator created successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create RAG answer generator: {e}")
            self.rag_answer_generator = None
            return False
    
    def is_ready(self) -> bool:
        """Check if all models are ready."""
        return (self.embedding_model is not None and 
                self.text_generation_model is not None and
                self.rag_answer_generator is not None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "embedding_model": self.embedding_model is not None,
            "summarizing_model": self.summarizing_model is not None,
            "reranker_model": self.reranker_model is not None,
            "text_generation_model": self.text_generation_model is not None,
            "rag_answer_generator": self.rag_answer_generator is not None,
            "total_models": sum([
                self.embedding_model is not None,
                self.summarizing_model is not None,
                self.reranker_model is not None,
                self.text_generation_model is not None,
                self.rag_answer_generator is not None
            ])
        }
    
    def get_rag_answer_generator(self) -> Optional[RAGAnswerGenerator]:
        """Get the RAG answer generator."""
        return self.rag_answer_generator


class DataProcessor(ComponentManager):
    """Manages data processing components."""
    
    def __init__(self, language: str = "en", embedding_model=None, summarizing_model=None):
        self.language = language
        self.embedding_model = embedding_model
        self.summarizing_model = summarizing_model
        self.logger = logger.bind(name="DataProcessor")
        
        # Components
        self.file_processor = None
        self.semantic_chunker = None
        self.mail_processor = None
    
    def initialize(self) -> bool:
        """Initialize data processing components."""
        try:
            self.logger.info("🔄 Initializing data processing components...")
            
            # Initialize file processor
            self.file_processor = MultiFormatProcessor(language=self.language)
            
            # Initialize semantic chunker
            self.semantic_chunker = SemanticChunker(
                language=self.language,
                file_processor=self.file_processor,
                embedding_model=self.embedding_model,
            )
            
            # Initialize mail processor
            self.mail_processor = MailJsonProcessor(
                language=self.language,
                summarizer=self.summarizing_model  # Will be set later
            )
            
            self.logger.info("✅ Data processing components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize data processing components: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if data processing components are ready."""
        return (self.file_processor is not None and 
                self.semantic_chunker is not None and
                self.mail_processor is not None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data processing statistics."""
        return {
            "file_processor": self.file_processor is not None,
            "semantic_chunker": self.semantic_chunker is not None,
            "mail_processor": self.mail_processor is not None
        }


class RAGManager(ComponentManager):
    """Manages RAG system components."""
    
    def __init__(self, language: str = "en", 
                 vector_store_type: str = "faiss",
                 search_engine_type: str = "bm25"):
        self.language = language
        self.vector_store_type = vector_store_type
        self.search_engine_type = search_engine_type
        self.logger = logger.bind(name="RAGManager")
        
        # RAG system
        self.rag_system = None
    
    def initialize(self, embedding_model, summarizing_model=None, reranker_model=None) -> bool:
        """Initialize RAG system."""
        try:
            self.logger.info("🔄 Initializing RAG system...")
            
            # Create RAG system
            self.rag_system = RAGSystem(
                language=self.language,
                vector_store_type=self.vector_store_type,
                search_engine_type=self.search_engine_type
            )
            
            # Initialize with models
            if not self.rag_system.initialize(
                embedding_model=embedding_model,
                summarizer=summarizing_model,
                reranker_model=reranker_model
            ):
                self.logger.error("❌ Failed to initialize RAG system")
                return False
            
            self.logger.info("✅ RAG system initialized")
            
            self.rag_system.load_data()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RAG system: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if RAG system is ready."""
        return self.rag_system is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if self.rag_system:
            return self.rag_system.get_system_stats()
        return {"status": "not_initialized"}


class PipelineOrchestrator:
    """Orchestrates the entire pipeline flow."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logger.bind(name="PipelineOrchestrator")
        
        # Component managers
        self.model_manager: Optional[ModelManager] = None
        self.data_processor: Optional[DataProcessor] = None
        self.rag_manager: Optional[RAGManager] = None
        
        # Pipeline state
        self.current_stage = PipelineStage.SCRAPING
        self.pipeline_stats = {}
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize the entire pipeline."""
        try:
            self.logger.info("🚀 Initializing Boost Pipeline...")
            
            # Initialize model manager
            self.model_manager = ModelManager(self.config.language)
            if not self.model_manager.initialize():
                self.logger.error("❌ Failed to initialize model manager")
                return False
            
            # Initialize data processor
            self.data_processor = DataProcessor(
                self.config.language, 
                self.model_manager.embedding_model,
                self.model_manager.summarizing_model
            )
            if not self.data_processor.initialize():
                self.logger.error("❌ Failed to initialize data processor")
                return False
            
            # Initialize RAG manager
            self.rag_manager = RAGManager(
                self.config.language,
                vector_store_type="faiss",
                search_engine_type="bm25"
            )
            if not self.rag_manager.initialize(
                embedding_model=self.model_manager.embedding_model,
                summarizing_model=self.model_manager.summarizing_model,
                reranker_model=self.model_manager.reranker_model
            ):
                self.logger.error("❌ Failed to initialize RAG manager")
                return False
            
            self.is_initialized = True
            self.logger.info("✅ Boost Pipeline initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline: {e}")
            return False
    
    def run_full_pipeline(self, chunk_files: List[str] = None, mail_files: List[str] = None) -> bool:
        """Run the complete pipeline."""
        try:
            if not self.is_initialized:
                self.logger.error("❌ Pipeline not initialized")
                return False
            
            self.logger.info("🚀 Starting full pipeline execution...")
            
            # Step 1: Process data (if not provided)
            if not chunk_files:
                self.current_stage = PipelineStage.PROCESSING
                success, chunk_file_list = self._process_data()
                if not success:
                    return False
                chunk_files = chunk_file_list
            
            # Step 2: Load data into RAG system
            self.current_stage = PipelineStage.EMBEDDING
            if not self.rag_manager.rag_system.load_data(chunk_files, mail_files):
                self.logger.error("❌ Failed to load data into RAG system")
                return False
            
            self.current_stage = PipelineStage.COMPLETE
            self.logger.info("✅ Full pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            return False
    
    def _process_data(self) -> Tuple[bool, List[str]]:
        """Process data files."""
        try:
            self.logger.info("🔄 Processing data files...")
            
            # Process mail JSON if present
            try:
                processed_files = self.data_processor.mail_processor.process_inputs()
                if processed_files:
                    self.logger.info(f"Processed {len(processed_files)} mail_json files")
            except Exception as exc:
                self.logger.warning(f"Mail JSON processing skipped: {exc}")
            
            # Process with semantic chunking
            success, chunk_file_list = self.data_processor.semantic_chunker.process_knowledge_base(
                max_files=self.config.max_files
            )
            
            if success:
                self.logger.info(f"✅ Data processing completed: {len(chunk_file_list)} files processed")
            else:
                self.logger.error("❌ Data processing failed")
            
            return success, chunk_file_list
            
        except Exception as e:
            self.logger.error(f"❌ Data processing error: {e}")
            return False, []
    
    def _create_embeddings(self, chunk_file_list: List[str]) -> bool:
        """Create embeddings from processed data."""
        try:
            self.logger.info("🔄 Creating embeddings...")
            
            success = self.rag_manager.rag_system.update_documents(chunk_file_list=chunk_file_list)
            
            if success:
                self.logger.info("✅ Embeddings created successfully")
            else:
                self.logger.error("❌ Embedding creation failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Embedding creation error: {e}")
            return False
    
    def _update_graph(self, chunk_file_list: List[str]) -> bool:
        """Update graph with processed data."""
        try:
            self.logger.info("🔄 Updating graph...")
            
            success = self.rag_manager.rag_system.update_graph(chunk_file_list=chunk_file_list)
            
            if success:
                self.logger.info("✅ Graph updated successfully")
            else:
                self.logger.error("❌ Graph update failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Graph update error: {e}")
            return False
    
    def query_system(self, question: str, max_sources: int = 5) -> Union[Dict[str, Any], str]:
        """Query the RAG system."""
        try:
            if not self.is_initialized:
                raise ValueError("Pipeline not initialized")
            
            response = self.rag_manager.rag_system.search(
                query=question, 
                max_results=max_sources
            )
            
            return response
                
        except Exception as e:
            self.logger.error(f"❌ Query failed: {e}")
            raise
    
    def load_data(self, chunk_files: List[str] = None, mail_files: List[str] = None) -> bool:
        """Load data into the RAG system."""
        if not self.is_initialized:
            self.logger.error("❌ Pipeline not initialized")
            return False
        
        return self.rag_manager.rag_system.load_data(chunk_files, mail_files)
    
    def update_data(self, chunk_files: List[str] = None, mail_files: List[str] = None) -> bool:
        """Update RAG system with new data."""
        if not self.is_initialized:
            self.logger.error("❌ Pipeline not initialized")
            return False
        
        return self.rag_manager.rag_system.update_data(chunk_files, mail_files)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "current_stage": self.current_stage.value,
            "model_manager": self.model_manager.get_stats() if self.model_manager else {},
            "data_processor": self.data_processor.get_stats() if self.data_processor else {},
            "rag_manager": self.rag_manager.get_stats() if self.rag_manager else {},
            "pipeline_stats": self.pipeline_stats
        }


class ImprovedBoostPipeline:
    """
    Improved Boost Pipeline with clean architecture.
    
    Features:
    - Clear separation of concerns
    - Modular component management
    - Easy testing and mocking
    - Flexible configuration
    - Better error handling
    """
    
    def __init__(self, config_path: str = "config/config.yaml", language: str = None):
        """Initialize the improved pipeline."""
        self.logger = logger.bind(name="ImprovedBoostPipeline")
        self.config_path = config_path
        
        # Load configuration
        self.config = load_config(config_path)
        if not self.config:
            raise ValueError(f"Failed to load configuration from {config_path}")
        
        self.language = get_config("language.default_language", "en") if language is None else language
        
        # Create pipeline configuration
        self.pipeline_config = PipelineConfig(
            language=self.language,
            max_files=get_config("pipeline.max_files"),
            max_depth=get_config("pipeline.max_depth", 2),
            delay=get_config("pipeline.delay", 1.0),
            source_url=get_config("pipeline.source_url"),
            output_dir=get_config("pipeline.output_dir")
        )
        
        # Create orchestrator
        self.orchestrator = PipelineOrchestrator(self.pipeline_config)
        
        self.logger.info("✅ Improved Boost Pipeline initialized")
    
    def initialize(self) -> bool:
        """Initialize the pipeline."""
        return self.orchestrator.initialize()
    
    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline."""
        return self.orchestrator.run_full_pipeline()
    
    def load_data(self, chunk_files: List[str] = None, mail_files: List[str] = None) -> bool:
        """Load data into the RAG system."""
        if not self.orchestrator.is_initialized:
            self.logger.error("❌ Pipeline not initialized")
            return False
        
        return self.orchestrator.rag_manager.rag_system.load_data(chunk_files, mail_files)
    
    def update_data(self, chunk_files: List[str] = None, mail_files: List[str] = None) -> bool:
        """Update RAG system with new data."""
        if not self.orchestrator.is_initialized:
            self.logger.error("❌ Pipeline not initialized")
            return False
        
        return self.orchestrator.rag_manager.rag_system.update_data(chunk_files, mail_files)
    
    def query_system(self, question: str, max_sources: int = 5) -> Union[Dict[str, Any], str]:
        """Query the RAG system."""
        return self.orchestrator.query_system(question, max_sources)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return self.orchestrator.get_pipeline_status()
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status."""
        if self.orchestrator.model_manager:
            return self.orchestrator.model_manager.get_stats()
        return {"status": "not_initialized"}


# Legacy compatibility wrapper
class BoostPipeline(ImprovedBoostPipeline):
    """Legacy compatibility wrapper for existing code."""
    
    def __init__(self, config_path: str = "config/config.yaml", language: str = None):
        super().__init__(config_path, language)
        self.logger = logger.bind(name="BoostPipeline")
    
    def setup_directories(self) -> bool:
        """Setup required directories."""
        try:
            directories = [
                get_config("rag.source_data.processed_data_path", "data/source_data/processed"),
                get_config("rag.source_data.new_data_path", "data/source_data/new"),
                get_config("rag.processed_data.raw_data_path", "data/processed/raw"),
                get_config("rag.processed_data.chunked_data_path", "data/processed/chunked"),
                get_config("rag.processed_data.embeddings_data_path", "data/processed/embeddings"),
                "logs",
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Directories setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up directories: {e}")
            return False
    
    def scrape_website(self, source_url: str = None, output_dir: str = None, 
                      max_depth: int = 2, delay: float = 1.0) -> bool:
        """Scrape website content."""
        try:
            if source_url is None:
                source_url = get_config('test_url_to_scrape')
            if output_dir is None:
                output_dir = get_config('data.source_data.new_data_path')
            
            output_dir = f"{output_dir}/{self.language}"
            
            self.logger.info(f"Starting data scraping from: {source_url}")
            
            from data_scraping.homepage_to_docs import HomepageToDocsConverter
            
            converter = HomepageToDocsConverter(
                output_dir=output_dir,
                delay=delay,
                max_depth=max_depth
            )
            
            summary = converter.convert_homepage(source_url)
            
            if summary['total_processed'] > 0:
                self.logger.info(f"Scraping completed: {summary['total_processed']} pages processed")
                return True
            else:
                self.logger.error("No pages were processed during scraping")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            return False
    
    def process_data(self, max_files: Optional[int] = None) -> Tuple[bool, List[str]]:
        """Process data files."""
        return self.orchestrator._process_data()
    
    def create_embeddings(self, chunk_file_list: List[str] = None, chunk_dir: str = None) -> bool:
        """Create embeddings from processed data."""
        return self.orchestrator._create_embeddings(chunk_file_list or [])
    
    def update_mail_hierarchy_graph(self, message_by_thread_file_list: List[str] = None, 
                                   message_by_thread_dir: str = None) -> bool:
        """Update mail hierarchy graph."""
        try:
            if message_by_thread_file_list is None:
                if message_by_thread_dir is None:
                    message_by_thread_dir = get_config("rag.processed_data.message_by_thread_path", "data/processed/message_by_thread")
                    message_by_thread_dir = f"{message_by_thread_dir}/{self.language}"
                message_by_thread_file_list = [f for f in Path(message_by_thread_dir).rglob("*_thread_*.json")]
            
            self.orchestrator.rag_manager.rag_system.update_data(mail_files=message_by_thread_file_list)
            return True
        except Exception as e:
            self.logger.error(f"Error during mail hierarchy graph update: {e}")
            return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Boost Pipeline - Unified Scalable Pipeline")
    parser.add_argument("--source-url", default="https://boost.ac.cn/doc/libs/latest/doc/html/boost_asio.html", help="Source URL to scrape")
    parser.add_argument("--input-dir", help="Input directory for processing")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--question", default="What is the main topic of the boost asio library?", help="Question to ask")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum depth for scraping")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    parser.add_argument("--max-files", default=5, type=int, help="Maximum number of files to process")
    parser.add_argument("--max-results", type=int, help="Maximum results for queries")
    parser.add_argument("--multi-step", action="store_true", help="Use multi-step reasoning")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate response")
    parser.add_argument("--language", default="en", help="Language for the pipeline")
    
    args = parser.parse_args()
    args.action = "evaluation"
    
    # Initialize pipeline
    pipeline = BoostPipeline(language=args.language)

    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        exit(1)
    
    success = False
    
    if args.action == "scrape":
        success = pipeline.scrape_website(args.source_url, args.output_dir, args.max_depth, args.delay)
        
    elif args.action == "process":
        success = pipeline.process_data(max_files=args.max_files)
        
    elif args.action == "embed":
        success = pipeline.create_embeddings(chunk_dir=args.input_dir)
    elif args.action == "mail_graph":
        success = pipeline.update_mail_hierarchy_graph(message_by_thread_dir=args.input_dir)
    elif args.action == "graph":
        success = pipeline.orchestrator.rag_manager.rag_system.update_graph(chunk_dir=args.input_dir)
        
    elif args.action == "full":
        success = pipeline.run_full_pipeline()
        
    elif args.action == "query":
        if not args.question:
            print("❌ --question is required for querying")
            exit(1)
    elif args.action == "evaluation":
        try:
            evaluator = SimpleRAGEvaluator(pipeline.orchestrator.rag_manager.rag_system)
            evaluator.evaluate_dataset()
            return True
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            exit(1)
        
    elif args.action == "status":
        status = pipeline.get_pipeline_status()
        print("📊 Pipeline Status:")
        print(f"  Config loaded: {status.get('config_loaded', False)}")
        print(f"  RAG System: {status.get('rag_system_type', 'Unknown')}")
        print(f"  RAG System initialized: {status.get('rag_system_initialized', False)}")
        
        if 'rag_statistics' in status:
            rag_stats = status['rag_statistics']
            print(f"  RAG system: {rag_stats.get('total_chunks', 0)} chunks loaded")
        
        success = True
    
    if success:
        print("✅ Operation completed successfully!")
    else:
        print("❌ Operation failed!")
        exit(1)


if __name__ == "__main__":
    main()
