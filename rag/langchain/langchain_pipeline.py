"""
LangChain-based RAG Pipeline Implementation
Implements the main ideas of the Boost Chatbot project using LangChain components.

Features:
1. Source data from data/processed/raw
2. Semantic chunking
3. Configurable embedding models from config.yaml
4. Hybrid retrieval (sparse, dense, graph, hierarchical) + reranker
5. Configurable QA LLM from config.yaml
6. Chroma vector database
7. Neo4j graph database
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import get_config
from data_processor.semantic_chunker import SemanticChunker
from rag.hybrid_retriever import HierarchicalRAG, GraphRAG
from rag.text_generation_model import TextGenerationModelFactory


@dataclass
class LangChainRAGConfig:
    """Configuration for LangChain RAG Pipeline."""
    # Data paths
    raw_data_path: str = "data/processed/raw"
    chunked_data_path: str = "data/processed/chunked"
    embeddings_path: str = "data/processed/embeddings"
    graph_cache_path: str = "data/processed/graph_cache"
    hierarchical_cache_path: str = "data/processed/hierarchical_cache"
    
    # Model configurations
    embedding_model: str = "gemma"
    llm_type: str = "ollama"
    llm_model: str = "gemma3:1b"
    language: str = "en"
    
    # Retrieval configuration
    top_k: int = 10
    similarity_threshold: float = 0.7
    use_reranker: bool = True
    
    # Neo4j configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"


class CustomLLM(LLM):
    """Custom LLM wrapper for our text generation models."""
    
    def __init__(self, text_generation_model):
        super().__init__()
        self.text_generation_model = text_generation_model
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Generate response using our custom text generation model."""
        return self.text_generation_model.generate_response(prompt, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "custom"


class LangChainRAGPipeline:
    """LangChain-based RAG Pipeline implementing the main project ideas."""
    
    def __init__(self, config: LangChainRAGConfig = None):
        """Initialize the LangChain RAG Pipeline."""
        self.logger = logger.bind(name="LangChainRAGPipeline")
        self.config = config or LangChainRAGConfig()
        
        # Initialize components
        self.embedding_model = None
        self.vectorstore = None
        self.graph = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        
        # Custom components
        self.semantic_chunker = None
        self.hierarchical_rag = None
        self.graph_rag = None
        self.bm25_retriever = None
        
        self.logger.info("🚀 LangChain RAG Pipeline initialized")
    
    def initialize_embedding_model(self):
        """Initialize embedding model from config."""
        self.logger.info(f"🔄 Initializing embedding model: {self.config.embedding_model}")
        
        try:
            # Get embedding configuration
            embedding_config = get_config(f"rag.embedding.{self.config.embedding_model}")
            model_name = embedding_config["model_name"]
            embedding_dimension = embedding_config["embedding_dimension"]
            
            # Initialize HuggingFace embeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.logger.info(f"✅ Embedding model initialized: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize embedding model: {e}")
            return False
    
    def initialize_semantic_chunker(self):
        """Initialize semantic chunker."""
        self.logger.info("🔄 Initializing semantic chunker...")
        
        try:
            # Create a shared embedding model for semantic chunker
            from data_processor.document_embedding import DocumentEmbedder
            shared_embedding_model = DocumentEmbedder(model_nick_name=self.config.embedding_model)
            
            self.semantic_chunker = SemanticChunker(
                language=self.config.language,
                embedding_model=shared_embedding_model
            )
            
            self.logger.info("✅ Semantic chunker initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize semantic chunker: {e}")
            return False
    
    def load_and_chunk_documents(self) -> List[Document]:
        """Load documents from raw data and perform semantic chunking."""
        self.logger.info("📚 Loading and chunking documents...")
        
        try:
            # Load documents from raw data path
            raw_path = Path(self.config.raw_data_path) / self.config.language
            documents = []
            
            # Load markdown files
            for md_file in raw_path.rglob("*.md"):
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create LangChain Document
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': str(md_file),
                            'file_type': 'markdown',
                            'language': self.config.language
                        }
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {md_file}: {e}")
            
            self.logger.info(f"📊 Loaded {len(documents)} documents")
            
            # Perform semantic chunking
            if self.semantic_chunker:
                self.logger.info("🔧 Performing semantic chunking...")
                chunked_docs = []
                
                for doc in documents:
                    # Convert to format expected by semantic chunker
                    chunk_data = {
                        'text': doc.page_content,
                        'metadata': doc.metadata
                    }
                    
                    # Get semantic chunks
                    chunks = self.semantic_chunker.chunk_text(chunk_data['text'], chunk_data['metadata'])
                    
                    # Convert to LangChain Documents
                    for chunk in chunks:
                        chunked_doc = Document(
                            page_content=chunk['text'],
                            metadata={
                                **doc.metadata,
                                'chunk_id': chunk['chunk_id'],
                                'chunk_type': chunk['metadata'].get('chunk_type', 'text'),
                                'semantic_similarity': chunk['metadata'].get('semantic_similarity', 0.0)
                            }
                        )
                        chunked_docs.append(chunked_doc)
                
                self.logger.info(f"✅ Created {len(chunked_docs)} semantic chunks")
                return chunked_docs
            
            else:
                # Fallback to recursive character splitting
                self.logger.info("🔧 Using recursive character splitting as fallback...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                chunked_docs = text_splitter.split_documents(documents)
                self.logger.info(f"✅ Created {len(chunked_docs)} chunks using recursive splitting")
                return chunked_docs
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load and chunk documents: {e}")
            return []
    
    def initialize_vectorstore(self, documents: List[Document]):
        """Initialize Chroma vector store."""
        self.logger.info("🔄 Initializing Chroma vector store...")
        
        try:
            # Create Chroma vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=f"{self.config.embeddings_path}/{self.config.embedding_model}/{self.config.language}/chroma"
            )
            
            self.logger.info("✅ Chroma vector store initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vector store: {e}")
            return False
    
    def initialize_neo4j_graph(self):
        """Initialize Neo4j graph database."""
        self.logger.info("🔄 Initializing Neo4j graph...")
        
        try:
            self.graph = Neo4jGraph(
                url=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
            
            self.logger.info("✅ Neo4j graph initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Neo4j graph: {e}")
            return False
    
    def initialize_retrievers(self, documents: List[Document]):
        """Initialize all retrieval components."""
        self.logger.info("🔄 Initializing retrieval components...")
        
        try:
            # 1. Dense retriever (vector store)
            dense_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.top_k}
            )
            
            # 2. Sparse retriever (BM25)
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.config.top_k
            
            # 3. Graph retriever (Neo4j)
            graph_retriever = self.graph.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.top_k}
            )
            
            # 4. Hierarchical retriever (custom)
            self.hierarchical_rag = HierarchicalRAG(
                llm_manager=self.llm,
                embedder=self.embedding_model,
                language=self.config.language
            )
            
            # Build hierarchical index
            chunk_data = [{'text': doc.page_content, 'metadata': doc.metadata} for doc in documents]
            self.hierarchical_rag.build_hierarchical_index(chunk_data)
            
            # 5. Ensemble retriever combining all methods
            self.retriever = EnsembleRetriever(
                retrievers=[dense_retriever, self.bm25_retriever, graph_retriever],
                weights=[0.5, 0.2, 0.3]  # Dense, Sparse, Graph
            )
            
            self.logger.info("✅ All retrieval components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize retrievers: {e}")
            return False
    
    def initialize_llm(self):
        """Initialize LLM from config."""
        self.logger.info(f"🔄 Initializing LLM: {self.config.llm_type}")
        
        try:
            # Get LLM configuration
            llm_config = get_config(f"rag.llm.{self.config.llm_type}")
            model_name = llm_config["model_name"]
            
            # Create text generation model using our factory
            text_generation_model = TextGenerationModelFactory.create_model(
                model_type=self.config.llm_type,
                model_name=model_name,
                config=llm_config
            )
            
            # Initialize the model
            if text_generation_model.initialize():
                # Wrap in custom LLM for LangChain compatibility
                self.llm = CustomLLM(text_generation_model)
                self.logger.info(f"✅ LLM initialized: {model_name}")
                return True
            else:
                self.logger.error("❌ Failed to initialize text generation model")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM: {e}")
            return False
    
    def initialize_qa_chain(self):
        """Initialize QA chain."""
        self.logger.info("🔄 Initializing QA chain...")
        
        try:
            # Create prompt template
            prompt_template = """
            You are a helpful assistant for C++ Boost library. 
            Use the following context to answer the question.
            If you don't know the answer based on the context, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            self.logger.info("✅ QA chain initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize QA chain: {e}")
            return False
    
    def initialize_pipeline(self):
        """Initialize the complete RAG pipeline."""
        self.logger.info("🚀 Initializing LangChain RAG Pipeline...")
        
        try:
            # 1. Initialize embedding model
            if not self.initialize_embedding_model():
                return False
            
            # 2. Initialize semantic chunker
            if not self.initialize_semantic_chunker():
                return False
            
            # 3. Load and chunk documents
            documents = self.load_and_chunk_documents()
            if not documents:
                self.logger.error("❌ No documents loaded")
                return False
            
            # 4. Initialize vector store
            if not self.initialize_vectorstore(documents):
                return False
            
            # 5. Initialize Neo4j graph
            if not self.initialize_neo4j_graph():
                return False
            
            # 6. Initialize LLM
            if not self.initialize_llm():
                return False
            
            # 7. Initialize retrievers
            if not self.initialize_retrievers(documents):
                return False
            
            # 8. Initialize QA chain
            if not self.initialize_qa_chain():
                return False
            
            self.logger.info("🎉 LangChain RAG Pipeline initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline."""
        self.logger.info(f"🔍 Processing query: {question}")
        
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Extract information
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Process source documents
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown")
                })
            
            # Get hierarchical results if available
            hierarchical_results = []
            if self.hierarchical_rag:
                try:
                    hierarchical_results = self.hierarchical_rag.search_hierarchy(question, top_k=3)
                except Exception as e:
                    self.logger.warning(f"⚠️ Hierarchical search failed: {e}")
            
            response = {
                "answer": answer,
                "sources": sources,
                "hierarchical_results": [
                    {
                        "text": result.text,
                        "score": result.score,
                        "metadata": result.metadata,
                        "hierarchy_level": result.hierarchy_level.value if hasattr(result, 'hierarchy_level') else "unknown"
                    }
                    for result in hierarchical_results
                ],
                "total_sources": len(sources),
                "model_info": {
                    "embedding_model": self.config.embedding_model,
                    "llm_type": self.config.llm_type,
                    "language": self.config.language
                }
            }
            
            self.logger.info(f"✅ Query processed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Query failed: {e}")
            return {
                "answer": "Sorry, I encountered an error processing your query.",
                "sources": [],
                "hierarchical_results": [],
                "error": str(e)
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of the pipeline components."""
        return {
            "embedding_model": self.config.embedding_model,
            "llm_type": self.config.llm_type,
            "language": self.config.language,
            "vectorstore_initialized": self.vectorstore is not None,
            "graph_initialized": self.graph is not None,
            "retriever_initialized": self.retriever is not None,
            "llm_initialized": self.llm is not None,
            "qa_chain_initialized": self.qa_chain is not None,
            "semantic_chunker_initialized": self.semantic_chunker is not None,
            "hierarchical_rag_initialized": self.hierarchical_rag is not None
        }


def create_langchain_pipeline(config: LangChainRAGConfig = None) -> LangChainRAGPipeline:
    """Factory function to create and initialize a LangChain RAG Pipeline."""
    pipeline = LangChainRAGPipeline(config)
    
    if pipeline.initialize_pipeline():
        logger.info("🎉 LangChain RAG Pipeline created successfully!")
        return pipeline
    else:
        logger.error("❌ Failed to create LangChain RAG Pipeline")
        return None


if __name__ == "__main__":
    # Example usage
    config = LangChainRAGConfig(
        embedding_model="gemma",
        llm_type="ollama",
        llm_model="gemma3:1b",
        language="en"
    )
    
    pipeline = create_langchain_pipeline(config)
    
    if pipeline:
        # Test query
        result = pipeline.query("What is Boost.Asio?")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])}")
        
        # Get status
        status = pipeline.get_pipeline_status()
        print(f"Pipeline Status: {status}")
