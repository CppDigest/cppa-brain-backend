"""
FastAPI-based REST API for C++ Boost Assistant.
Provides endpoints for scraping, processing, querying, and managing the knowledge base.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from pathlib import Path


# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main_pipeline import ImprovedBoostPipeline
from rag.improved_rag_system import RAGSystem, SearchRequest, SearchMethod, SearchScope
from utils.config import get_config
from chat_history_manager import chat_manager, ChatMessage


# Pydantic models for API requests/responses
class RagSettingRequest(BaseModel):
    """RAG setting request"""
    embedding: str = Field(None, description="Embedding type")
    database: str = Field(None, description="Database type")
    retrieval_weights: Optional[Dict[str, float]] = Field(None, description="Retrieval method weights")
    reranker: str = Field(None, description="Reranker type")
    context_filtering: str = Field(None, description="Context filtering type")
    llm: str = Field(None, description="LLM type")
    evaluation: str = Field(None, description="Evaluation type")
    language: str = Field(None, description="Language for the RAG system")


class ScrapeRequest(BaseModel):
    source_url: str = Field(..., description="URL to scrape")
    max_depth: int = Field(2, description="Maximum depth for following links")
    delay: float = Field(1.0, description="Delay between requests in seconds")
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    use_enhanced_rag: bool = Field(False, description="Use enhanced RAG system")


class UpdateRequest(BaseModel):
    new_source_url: str = Field(..., description="New URL to scrape")
    max_depth: int = Field(2, description="Maximum depth for following links")
    delay: float = Field(1.0, description="Delay between requests in seconds")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    max_results: int = Field(5, description="Maximum number of results to return")
    use_enhanced_rag: bool = Field(False, description="Use enhanced RAG system")
    use_multi_step: bool = Field(False, description="Use multi-step reasoning (enhanced only)")
    use_evaluation: bool = Field(False, description="Evaluate response (enhanced only)")
    use_chat_history: bool = Field(True, description="Use chat history for context")
    clear_history: bool = Field(False, description="Clear chat history before processing")
    client_id: str = Field("default_client", description="Client ID for chat history management")
    retrieval_weights: Optional[Dict[str, float]] = Field(None, description="Retrieval method weights")
    embedding: Optional[str] = Field(None, description="Embedding model to use")
    database: Optional[str] = Field(None, description="Database type to use")
    llm: Optional[str] = Field(None, description="LLM model to use")
    language: Optional[str] = Field(None, description="Language for the RAG system")


class ChannelRequest(BaseModel):
    channel_id: str = Field(..., description="Channel ID")
    channel_name: Optional[str] = Field(None, description="Channel name")


class ChannelResponse(BaseModel):
    channel_id: str
    channel_name: str
    message_count: int
    last_activity: str
    created_at: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_results: Optional[List[Dict[str, Any]]] = Field(None, description="Raw retrieval results used for answer generation")
    timestamp: str
    chat_history: Optional[List[Dict[str, Any]]] = Field(None, description="Recent chat history")
    history_length: int = Field(0, description="Number of entries in chat history")
    embedding_model: Optional[str] = Field(None, description="Name of the embedding model used")


class StatusResponse(BaseModel):
    pipeline_stats: Dict[str, Any]
    config_loaded: bool
    components_initialized: Dict[str, bool]
    rag_statistics: Optional[Dict[str, Any]]


class ModelInfo(BaseModel):
    name: str
    size: int
    parameter_size: str


class LLMModelsResponse(BaseModel):
    models: List[ModelInfo]
    total_count: int
    sorted_by: str = "size (smallest first)"


class EvaluationRequest(BaseModel):
    dataset_path: str = Field("data/validation_dataset.json", description="Path to validation dataset")
    max_questions: int = Field(5, description="Maximum number of questions to evaluate")


class EvaluationResponse(BaseModel):
    total_questions: int
    average_similarity: float
    average_time: float
    accuracy: float
    good_answers: int
    results: List[Dict[str, Any]]


class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Maximum number of results to return")
    retrieval_weights: Optional[Dict[str, float]] = Field(None, description="Retrieval method weights")
    embedding: Optional[str] = Field(None, description="Embedding model to use")
    database: Optional[str] = Field(None, description="Database type to use")
    language: Optional[str] = Field(None, description="Language for the RAG system")


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time: float
    timestamp: str


class ThreadInfo(BaseModel):
    url: str = Field(..., description="Thread URL")
    thread_id: str = Field(..., description="Thread ID")
    subject: str = Field(..., description="Thread subject")
    date_active: str = Field(..., description="Thread active date")
    starting_email: str = Field(..., description="Starting email URL")
    emails_url: str = Field(..., description="Emails URL")
    replies_count: int = Field(0, description="Number of replies")
    votes_total: int = Field(0, description="Total votes")


class MessageData(BaseModel):
    message_id: str = Field(..., description="Unique message ID")
    subject: str = Field(..., description="Message subject")
    content: str = Field(..., description="Message content")
    thread_url: str = Field(..., description="Thread URL")
    parent: Optional[str] = Field(None, description="Parent message URL")
    children: List[str] = Field(default_factory=list, description="Children message URLs")
    sender_address: str = Field(..., description="Sender email address")
    from_field: str = Field(..., description="From field")
    date: str = Field(..., description="Message date")
    to: str = Field(..., description="To field")
    cc: str = Field(default="", description="CC field")
    reply_to: str = Field(default="", description="Reply-To field")
    url: str = Field(..., description="Message URL")


class EmailThreadRequest(BaseModel):
    timestamp: str = Field(..., description="Request timestamp")
    requestId: str = Field(..., description="Unique request identifier")
    thread_info: ThreadInfo = Field(..., description="Thread information")
    messages: List[MessageData] = Field(..., description="List of messages")
    message_count: int = Field(..., description="Total message count")


class EmailMessagesRequest(BaseModel):
    timestamp: str = Field(..., description="Request timestamp")
    requestId: str = Field(..., description="Unique request identifier")
    messages: List[MessageData] = Field(..., description="List of messages")
    message_count: int = Field(..., description="Total message count")


class UpdateMessageRequest(BaseModel):
    message_id: str = Field(..., description="Message ID to update")
    subject: Optional[str] = Field(None, description="Updated subject")
    content: Optional[str] = Field(None, description="Updated content")
    sender_address: Optional[str] = Field(None, description="Updated sender address")
    from_field: Optional[str] = Field(None, description="Updated from field")
    date: Optional[str] = Field(None, description="Updated date")
    to: Optional[str] = Field(None, description="Updated to field")
    cc: Optional[str] = Field(None, description="Updated CC field")
    reply_to: Optional[str] = Field(None, description="Updated Reply-To field")
    url: Optional[str] = Field(None, description="Updated URL")


class DeleteMessageRequest(BaseModel):
    message_id: str = Field(..., description="Message ID to delete")


class NewThreadRequest(BaseModel):
    thread_id: str = Field(..., description="Unique thread ID")
    subject: str = Field(..., description="Thread subject")
    url: Optional[str] = Field(None, description="Thread URL")
    date_active: Optional[str] = Field(None, description="Thread active date")
    starting_email: Optional[str] = Field(None, description="Starting email ID")
    emails_url: Optional[str] = Field(None, description="Emails URL")
    replies_count: int = Field(0, description="Number of replies")
    votes_total: int = Field(0, description="Total votes")


class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Dict[str, Any]
    timestamp: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: Dict[str, Any]
    timestamp: str


class ActionStatsResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Boost Knowledge Assistant API",
    description=""""
    The **Boost Assistant API** provides an intelligent interface for exploring and understanding the C++ Boost library.
    
    It integrates a Retrieval-Augmented Generation (RAG) backend capable of:
    - Retrieving relevant documentation and code examples
    - Generating context-aware, accurate answers using an instruction-tuned LLM
    - Dynamically updating its knowledge base with new documents or user-provided data
    
    This API is ideal for building interactive Boost documentation assistants or developer support chatbots.
    """,
    version="2.0.0"
)

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Global pipeline instances
main_pipeline: Optional[ImprovedBoostPipeline] = None
rag_system: Optional[RAGSystem] = None

# Simple in-memory channel storage (in production, use a proper database)
channel_storage: Dict[str, Dict[str, Any]] = {}

# Cached model lists
cached_llm_models: List[Dict[str, Any]] = []
cached_embedding_models: List[str] = []

@app.on_event("startup")
async def startup_event():
    """Initialize the pipelines on startup."""
    global main_pipeline, rag_system, cached_llm_models, cached_embedding_models
    try:
        # Create improved pipeline
        main_pipeline = ImprovedBoostPipeline()
        
        # Initialize pipeline
        if not main_pipeline.initialize():
            raise Exception("Failed to initialize improved pipeline")
        rag_system = main_pipeline.orchestrator.rag_manager.rag_system
        
        # Load and cache Ollama models at startup
        try:
            from text_generation.ollama_config import get_llm_models_sorted_by_size
            cached_llm_models = get_llm_models_sorted_by_size()
            logger.info(f"Loaded and cached {len(cached_llm_models)} LLM models at startup")
        except Exception as e:
            logger.warning(f"Failed to load Ollama models at startup: {e}")
            cached_llm_models = []
        
        # Load and cache embedding models at startup
        try:
            cached_embedding_models = main_pipeline.orchestrator.model_manager.get_available_embedding_models()
            logger.info(f"Loaded and cached {len(cached_embedding_models)} embedding models at startup")
        except Exception as e:
            logger.warning(f"Failed to load embedding models at startup: {e}")
            cached_embedding_models = []
        
        logger.info("VectorDataBuild API started successfully with dual RAG support")
    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}")
        raise

@app.get("/doc/fulltext")
async def get_document_fulltext(path: str, language: str = None):
    """Return full text for a processed document by relative source path.
    Looks under data/source_data/processed/{lang}/.
    """
    try:
        lang = language or get_config("language.default_language", "en")
        base = Path(get_config("data.source_data.processed_data_path", "data/source_data/processed")) / lang
        # path comes like 'asio/scraped/boost_asio/overview.md'
        safe_path = Path(path.replace("\\", "/"))
        full_path = (base / safe_path).resolve()
        if base.resolve() not in full_path.parents and base.resolve() != full_path.parent.resolve():
            raise HTTPException(status_code=400, detail="Invalid path")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return {"path": str(safe_path), "content": full_path.read_text(encoding="utf-8", errors="ignore")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading fulltext: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/processed", response_model=Dict[str, Any])
async def index_processed(language: str = None, max_files: int = None):
    """Index documents from data/source_data/processed/{lang} and load into RAG.
    - Runs semantic chunking over processed corpus (non-destructive)
    - Loads chunks into vector/BM25/graph indices
    """
    try:
        if main_pipeline is None:
            raise HTTPException(status_code=503, detail="Pipelines not initialized")
        lang = language or get_config("language.default_language", "en")
        # Run chunking using processed input dir
        processed_dir = Path(get_config("data.source_data.processed_data_path", "data/source_data/processed")) / lang
        if not processed_dir.exists():
            raise HTTPException(status_code=404, detail=f"Processed directory not found: {processed_dir}")
        # Access semantic_chunker via orchestrator's data processor
        chunker = main_pipeline.orchestrator.data_processor.semantic_chunker
        success, chunk_files = chunker.process_knowledge_base(
            raw_file_list=None,
            max_files=max_files,
            input_dir=str(processed_dir)
        )
        if not success:
            raise HTTPException(status_code=500, detail="Chunking failed")
        # Load data into RAG system
        loaded = main_pipeline.orchestrator.rag_manager.rag_system.load_data(
            chunk_files=[str(p) for p in chunk_files]
        )
        if not loaded:
            raise HTTPException(status_code=500, detail="Failed to load data into RAG system")
        return {
            "message": "Indexing completed",
            "language": lang,
            "chunk_files": [str(p) for p in chunk_files],
            "total_chunk_files": len(chunk_files)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error indexing processed data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mail/message")
async def get_mail_message(url: str = None, message_id: str = None):
    """Return detailed mail message fields by URL or message_id (hash).
    Requires the mail hierarchy graph to be loaded in memory.
    """
    try:
        if rag_system is None or rag_system.graph_manager is None or rag_system.graph_manager.mail_hierarchy is None:
            raise HTTPException(status_code=503, detail="Mail hierarchy not available")
        mh = rag_system.graph_manager.mail_hierarchy
        # Build lookup by url or by node id
        node_id = None
        if message_id and message_id in mh.graph:
            node_id = message_id
        elif url:
            for n in mh.graph.nodes:
                if mh.graph.nodes[n].get("url") == url:
                    node_id = n
                    break
        if not node_id:
            raise HTTPException(status_code=404, detail="Message not found")
        data = mh.graph.nodes[node_id]
        return {
            "message_id": data.get("message_id"),
            "sender_address": data.get("sender_address") or data.get("from"),
            "date": data.get("date"),
            "to": data.get("to"),
            "cc": data.get("cc"),
            "reply_to": data.get("reply_to"),
            "subject": data.get("subject"),
            "url": data.get("url"),
            "content": mh._create_node_text(data)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching mail message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_pipeline")
async def set_pipeline(rag_setting: RagSettingRequest):
    """Set RAG pipeline configuration."""
    
    global main_pipeline
    try:
        if main_pipeline is None:
            raise HTTPException(status_code=503, detail="Pipelines not initialized")
        
        # Initialize RAG system with new settings
        success = main_pipeline.reset_properties(
            llm_group=rag_setting.llm, 
            embedding=rag_setting.embedding, 
            language=rag_setting.language
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to initialize RAG system with new settings")
        
        logger.info(f"RAG system updated with settings: {rag_setting.dict()}")
        return {
            "message": "RAG pipeline settings updated successfully",
            "settings": rag_setting.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update RAG pipeline settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    # return {
    #     "message": "VectorDataBuild API",
    #     "status": "running",
    #     "docs": "/docs"
    # }
    return FileResponse(get_config("start_page", "src/templates/index.html"))


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    if main_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/chat/clear")
async def clear_chat_history():
    """
    Clear the chat history.
    """
    if main_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    try:
        if main_pipeline.rag_system:
            main_pipeline.rag_system.clear_chat_history()
            return {"message": "Chat history cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="RAG system not available")
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
async def get_chat_history(include_metadata: bool = False, channel_id: Optional[str] = None):
    """
    Get the current chat history for a specific channel.
    """
    if main_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    try:
        if channel_id and channel_id in channel_storage:
            # Return channel-specific history
            channel = channel_storage[channel_id]
            return {
                "chat_history": channel.get("messages", []),
                "history_length": len(channel.get("messages", [])),
                "channel_id": channel_id,
                "channel_name": channel.get("name", "Unknown")
            }
        elif main_pipeline.rag_system:
            # Return global chat history
            history = main_pipeline.rag_system.get_chat_history(include_metadata=include_metadata)
            return {
                "chat_history": history,
                "history_length": len(history),
                "max_history_length": main_pipeline.rag_system.max_history_length
            }
        else:
            raise HTTPException(status_code=404, detail="RAG system not available")
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/channels", response_model=ChannelResponse)
async def create_channel(request: ChannelRequest):
    """
    Create a new chat channel.
    """
    try:
        channel_id = request.channel_id
        channel_name = request.channel_name or f"Channel {channel_id}"
        
        if channel_id in channel_storage:
            raise HTTPException(status_code=400, detail="Channel already exists")
        
        channel_storage[channel_id] = {
            "name": channel_name,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        
        return ChannelResponse(
            channel_id=channel_id,
            channel_name=channel_name,
            message_count=0,
            last_activity=datetime.now().isoformat(),
            created_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/channels", response_model=List[ChannelResponse])
async def list_channels():
    """
    List all available chat channels.
    """
    try:
        channels = []
        for channel_id, channel_data in channel_storage.items():
            channels.append(ChannelResponse(
                channel_id=channel_id,
                channel_name=channel_data.get("name", "Unknown"),
                message_count=len(channel_data.get("messages", [])),
                last_activity=channel_data.get("last_activity", ""),
                created_at=channel_data.get("created_at", "")
            ))
        
        # Sort by last activity
        channels.sort(key=lambda x: x.last_activity, reverse=True)
        return channels
    except Exception as e:
        logger.error(f"Error listing channels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/channels/{channel_id}")
async def delete_channel(channel_id: str):
    """
    Delete a chat channel.
    """
    try:
        if channel_id not in channel_storage:
            raise HTTPException(status_code=404, detail="Channel not found")
        
        del channel_storage[channel_id]
        return {"message": f"Channel {channel_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/channels/{channel_id}/messages")
async def add_message_to_channel(channel_id: str, message: Dict[str, Any]):
    """
    Add a message to a specific channel.
    """
    try:
        if channel_id not in channel_storage:
            raise HTTPException(status_code=404, detail="Channel not found")
        
        channel = channel_storage[channel_id]
        if "messages" not in channel:
            channel["messages"] = []
        
        message_data = {
            "type": message.get("type", "user"),
            "text": message.get("text", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        channel["messages"].append(message_data)
        channel["last_activity"] = datetime.now().isoformat()
        
        return {"message": "Message added successfully"}
    except Exception as e:
        logger.error(f"Error adding message to channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current pipeline status."""
    if main_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    try:
        # Get status from both pipelines
        basic_status = main_pipeline.get_pipeline_status()
        
        # Combine status information
        combined_status = {
            "pipeline_stats": basic_status.get("pipeline_stats", {}),
            "config_loaded": basic_status.get("config_loaded", False),
            "components_initialized": {**{f"basic_{k}": v for k, v in basic_status.get("components_initialized", {}).items()}},
            "rag_statistics": basic_status.get("rag_statistics")
        }
        
        return StatusResponse(**combined_status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{client_id}")
async def get_chat_history(client_id: str, max_messages: int = 20, include_metadata: bool = False):
    """Get chat history for a specific client."""
    try:
        history = chat_manager.get_chat_history(
            client_id=client_id,
            max_messages=max_messages,
            include_metadata=include_metadata
        )
        
        return {
            "client_id": client_id,
            "chat_history": history,
            "history_length": len(history),
            "max_messages": max_messages
        }
        
    except Exception as e:
        logger.error(f"Error getting chat history for {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/history/{client_id}")
async def clear_chat_history(client_id: str):
    """Clear chat history for a specific client."""
    try:
        success = chat_manager.clear_history(client_id)
        
        if success:
            return {"message": f"Chat history cleared for client {client_id}"}
        else:
            raise HTTPException(status_code=404, detail="Client not found")
            
    except Exception as e:
        logger.error(f"Error clearing chat history for {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/session/{client_id}")
async def get_session_stats(client_id: str):
    """Get session statistics for a specific client."""
    try:
        stats = chat_manager.get_session_stats(client_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting session stats for {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stats")
async def get_global_chat_stats():
    """Get global chat statistics."""
    try:
        stats = chat_manager.get_global_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting global chat stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/cleanup")
async def cleanup_expired_sessions():
    """Clean up expired chat sessions."""
    try:
        cleaned_count = chat_manager.cleanup_expired_sessions()
        return {
            "message": f"Cleaned up {cleaned_count} expired sessions",
            "cleaned_sessions": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up expired sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/llm", response_model=LLMModelsResponse)
async def get_llm_models():
    """Get available LLM models."""
    global cached_llm_models
    try:
        if not cached_llm_models:
            # Try to load models if not cached
            try:
                from text_generation.ollama_config import get_llm_models_sorted_by_size
                cached_llm_models = get_llm_models_sorted_by_size()
            except Exception as e:
                logger.warning(f"Failed to load LLM models: {e}")
                cached_llm_models = []
        
        models = []
        for model in cached_llm_models:
            models.append(ModelInfo(
                name=model.get('name', 'Unknown'),
                size=model.get('size', 0),
                parameter_size=model.get('parameter_size', 'Unknown')
            ))
        
        return LLMModelsResponse(
            models=models,
            total_count=len(models),
            sorted_by="size (smallest first)"
        )
    except Exception as e:
        logger.error(f"Error getting LLM models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/embedding")
async def get_embedding_models():
    """Get available embedding models."""
    global cached_embedding_models
    try:
        if not cached_embedding_models:
            # Try to load models if not cached
            try:
                if main_pipeline and main_pipeline.orchestrator:
                    cached_embedding_models = main_pipeline.orchestrator.model_manager.get_available_embedding_models()
                else:
                    cached_embedding_models = ["sentence-transformers/all-MiniLM-L6-v2"]
            except Exception as e:
                logger.warning(f"Failed to load embedding models: {e}")
                cached_embedding_models = ["sentence-transformers/all-MiniLM-L6-v2"]
        
        return {
            "models": cached_embedding_models,
            "total_count": len(cached_embedding_models),
            "default_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    except Exception as e:
        logger.error(f"Error getting embedding models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag-systems", response_model=Dict[str, Any])
async def get_rag_systems():
    """Get information about available RAG systems."""
    try:
        return {
            "available_systems": {
                "main": {
                    "name": "RAG System",
                    "description": "RAG with retrieval and generation",
                    "features": ["Semantic search", "Basic chunking", "Simple Q&A"],
                    "initialized": main_pipeline is not None
                },
            },
            "default_system": "main",
            "recommendations": "Use for simple Q&A, quick responses, basic documentation queries"
        }
    except Exception as e:
        logger.error(f"Error getting RAG systems info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrape", response_model=Dict[str, Any])
async def scrape_data(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Scrape data from a website
    """
    # Select appropriate pipeline based on RAG system type
        
    if main_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    try:
        # Run scraping in background
        def run_scraping():
            success = main_pipeline.run_full_pipeline(
                request.source_url,
                request.max_depth,
                request.delay,
                request.max_files
            )
            if not success:
                logger.error(f"Scraping failed for {request.source_url}")
        
        background_tasks.add_task(run_scraping)
        
        return {
            "message": "Scraping started",
            "source_url": request.source_url,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting scraping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Query the RAG system with a question.
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Handle chat history
        if request.clear_history:
            chat_manager.clear_history(request.client_id)
            logger.info(f"Cleared chat history for client: {request.client_id}")
        
        # Add user message to chat history
        chat_manager.add_message(
            client_id=request.client_id,
            role="user",
            content=request.question,
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # Get conversation context if using chat history
        context = ""
        if request.use_chat_history:
            context = chat_manager.get_conversation_context(request.client_id, max_messages=5)
        
        # Create enhanced query with context
        enhanced_query = request.question
        if context:
            enhanced_query = f"Context: {context}\n\nQuestion: {request.question}"
        
        # Create search request
        search_request = SearchRequest(
            query=enhanced_query,
            method=SearchMethod.HYBRID,
            scope=SearchScope.BOTH,
            max_results=request.max_results
        )
        
        # Perform search
        search_response = rag_system.orchestrator.search(search_request)
        
        # Prepare retrieval results for display (include metadata & type)
        retrieval_results = []
        for result in search_response.results:
            retrieval_results.append({
                "text": getattr(result, 'text', ''),
                "source_file": getattr(result, 'source_file', ''),
                "score": getattr(result, 'score', None),
                "retrieval_method": getattr(result, 'retrieval_method', ''),
                "source_type": getattr(result, 'source_type', ''),
                "metadata": getattr(result, 'metadata', {})
            })
        
        # Get RAG answer generator
        rag_answer_generator = main_pipeline.orchestrator.model_manager.get_rag_answer_generator()
        
        if rag_answer_generator is None or not rag_answer_generator.is_ready:
            # Fallback to simple answer if RAG generator not available
            answer_parts = []
            sources = []
            
            for i, result in enumerate(search_response.results):
                answer_parts.append(result.text[:200] + "...")
                sources.append({
                    "source_file": result.source_file,
                    "score": result.score,
                    "retrieval_method": result.retrieval_method
                })
            
            answer = "\n\n".join(answer_parts)
        else:
            # Use RAG answer generator for comprehensive answer
            # Generate comprehensive answer using RAG generator
            rag_response = rag_answer_generator.generate_answer(
                question=request.question,
                retrieval_results=retrieval_results,
                max_context_length=4000,
                include_sources=True
            )
            
            answer = rag_response["answer"]
            sources = rag_response["sources"]
        
        # Add assistant response to chat history
        chat_manager.add_message(
            client_id=request.client_id,
            role="assistant",
            content=answer,
            metadata={
                "sources_count": len(sources),
                "search_time": search_response.search_time,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Log retrieval results for debugging
        logger.info(f"📊 Retrieved {len(sources)} sources for query: '{request.question}'")
        
        # Count retrieval methods
        method_counts = {}
        for i, source in enumerate(sources):
            method = source.get('retrieval_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            logger.info(f"   Source {i+1}: {source.get('source_file', 'Unknown')} (score: {source.get('score', 0):.3f}, method: {method})")
        
        # Log retrieval method summary
        if method_counts:
            logger.info("📈 Retrieval method breakdown:")
            for method, count in method_counts.items():
                logger.info(f"   {method}: {count} sources")
        
        # Handle different answer formats
        answer_text = answer
        if isinstance(answer, dict):
            answer_text = answer.get('answer', str(answer))
        
        # Get chat history for response
        chat_history = None
        history_length = 0
        if request.use_chat_history:
            chat_history = chat_manager.get_chat_history(
                client_id=request.client_id,
                max_messages=10,
                include_metadata=False
            )
            history_length = len(chat_history)
        
        # Get embedding model information
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Default model
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            retrieval_results=retrieval_results,
            timestamp=datetime.now().isoformat(),
            chat_history=chat_history,
            history_length=history_length,
            embedding_model=embedding_model
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats", response_model=Dict[str, Any])
async def get_rag_statistics():
    """
    Get RAG system statistics.
    """
    if main_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    try:
        if rag_system:
            stats = rag_system.get_system_stats()
            return stats
        else:
            return {"message": "No RAG system loaded"}
    except Exception as e:
        logger.error(f"Error getting RAG statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/rag", response_model=Dict[str, Any])
async def get_rag_config():
    """
    Get RAG configuration options from config.yaml.
    """
    try:
        
        # Extract relevant configuration options
        rag_config = {
            "database_types": get_config("rag.database.db_types_list", ["faiss", "chroma"]),
            "retrieval_default_weight": get_config("rag.retrieval.retrieval_default_weight", 
                                                 {"vector search": 0.5, "bm25 search": 0.2, "graph search": 0.15, "hierarchical search": 0.15}),
            "language_types": get_config("language.language_types_list", ["en", "cn", "fr", "es", "de", "it", "pt", "ru", "zh", "ja", "ko"]),
            "defaults": {
                "database": get_config("rag.database.default_db_type", "faiss"),
                "language": get_config("language.default_language", "en")
            }
        }
        
        logger.info(f"Retrieved RAG configuration: {rag_config}")
        return rag_config
    except Exception as e:
        logger.error(f"Error getting RAG configuration: {e}")
        # Return fallback configuration on error
        return {
            "database_types": ["faiss", "chroma"],
            "retrieval_default_weight": {"vector search": 0.5, "bm25 search": 0.2, "graph search": 0.15, "hierarchical search": 0.15},
            "language_types": ["en", "cn", "fr", "es", "de", "it", "pt", "ru", "zh", "ja", "ko"],
            "defaults": {
                "database": "faiss",
                "language": "en"
            }
        }


@app.post("/rag/load", response_model=Dict[str, Any])
async def load_rag_system():
    """
    Load RAG system
    """
    if main_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    try:
        success = main_pipeline.orchestrator.rag_manager.rag_system.load_embedding_data()
        if success:     
            return {
                "message": f"RAG system loaded",
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to load RAG system")
    except Exception as e:
        logger.error(f"Error loading RAG system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_system(request: SearchQueryRequest):
    """
    Search the RAG system for relevant documents using retrieval only.
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Create search request
        search_request = SearchRequest(
            query=request.query,
            method=SearchMethod.HYBRID,
            scope=SearchScope.BOTH,
            max_results=request.max_results
        )
        
        # Perform search
        search_response = rag_system.orchestrator.search(search_request)
        
        # Format results
        results = []
        for result in search_response.results:
            results.append({
                "source_file": result.source_file,
                "score": result.score,
                "retrieval_method": result.retrieval_method,
                "text": result.text,
                "metadata": getattr(result, 'metadata', {})
            })
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time=search_response.search_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maillist/messages/new", response_model=SuccessResponse)
async def add_new_messages(request: EmailMessagesRequest):
    """
    Add new messages to mail hierarchical RAG database.
    """
    if main_pipeline is None or rag_system is None:
        raise HTTPException(
            status_code=503, 
            detail={
                "success": False,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "RAG system not initialized",
                    "details": "The mail hierarchy system is not available"
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        # Get mail hierarchical RAG system
        mail_hierarchy = rag_system.graph_manager.mail_hierarchy
        
        added_count = 0
        failed_messages = []
        
        for message_data in request.messages:
            try:
                # Create message data structure for the hierarchy
                message_dict = {
                    "message_id": message_data.message_id,
                    "thread_id": message_data.thread_url,
                    "subject": message_data.subject,
                    "content": message_data.content,
                    "sender_address": message_data.sender_address,
                    "from": message_data.from_field,
                    "date": message_data.date,
                    "to": message_data.to,
                    "cc": message_data.cc,
                    "reply_to": message_data.reply_to,
                    "url": message_data.url,
                    "parent": message_data.parent,
                    "children": message_data.children
                }
                
                # Add message to hierarchy
                success = mail_hierarchy.add_new_message(message_dict)
                if success:
                    added_count += 1
                else:
                    failed_messages.append(message_data.message_id)
                    
            except Exception as e:
                logger.error(f"Error adding message {message_data.message_id}: {e}")
                failed_messages.append(message_data.message_id)
        
        # Update graph and rebuild index if any messages were added
        if added_count > 0:
            mail_hierarchy.save_graph()
        
        return SuccessResponse(
            message=f"Successfully processed {added_count} out of {len(request.messages)} messages",
            data={
                "requestId": request.requestId,
                "added_count": added_count,
                "total_messages": len(request.messages),
                "failed_messages": failed_messages,
                "graph_nodes": len(mail_hierarchy.graph.nodes),
                "graph_edges": len(mail_hierarchy.graph.edges)
            },
            timestamp=datetime.now().isoformat()
        )
            
    except Exception as e:
        logger.error(f"Error adding new messages: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                },
                "timestamp": datetime.now().isoformat()
            }
        )


@app.put("/maillist/message/update", response_model=SuccessResponse)
async def update_message(request: UpdateMessageRequest):
    """
    Update existing message in mail hierarchical graph.
    """
    if main_pipeline is None or rag_system is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "RAG system not initialized",
                    "details": "The mail hierarchy system is not available"
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        # Get mail hierarchical RAG system
        mail_hierarchy = rag_system.graph_manager.mail_hierarchy
        
        # Check if message exists
        if request.message_id not in mail_hierarchy.graph:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": {
                        "code": "RESOURCE_NOT_FOUND",
                        "message": "Message not found",
                        "details": f"Message {request.message_id} does not exist"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Create update data (only include non-None fields)
        update_data = {}
        if request.subject is not None:
            update_data["subject"] = request.subject
        if request.content is not None:
            update_data["content"] = request.content
        if request.sender_address is not None:
            update_data["sender_address"] = request.sender_address
        if request.from_field is not None:
            update_data["from"] = request.from_field
        if request.date is not None:
            update_data["date"] = request.date
        if request.to is not None:
            update_data["to"] = request.to
        if request.cc is not None:
            update_data["cc"] = request.cc
        if request.reply_to is not None:
            update_data["reply_to"] = request.reply_to
        if request.url is not None:
            update_data["url"] = request.url
        
        # Update message in hierarchy
        success = mail_hierarchy.update_message_by_node(
            node_id=request.message_id,
            content=update_data
        )
        
        if success:
            # Save updated graph
            mail_hierarchy.save_graph()
            
            return SuccessResponse(
                message=f"Successfully updated message {request.message_id}",
                data={
                    "message_id": request.message_id,
                    "updated_fields": list(update_data.keys()),
                    "graph_nodes": len(mail_hierarchy.graph.nodes),
                    "graph_edges": len(mail_hierarchy.graph.edges)
                },
                timestamp=datetime.now().isoformat()
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": {
                        "code": "UPDATE_FAILED",
                        "message": "Failed to update message",
                        "details": f"Could not update message {request.message_id}"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating message: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                },
                "timestamp": datetime.now().isoformat()
            }
        )


@app.delete("/maillist/message/delete", response_model=SuccessResponse)
async def delete_message(request: DeleteMessageRequest):
    """
    Delete message from mail hierarchical graph.
    """
    if main_pipeline is None or rag_system is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "RAG system not initialized",
                    "details": "The mail hierarchy system is not available"
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        # Get mail hierarchical RAG system
        mail_hierarchy = rag_system.graph_manager.mail_hierarchy
        
        # Check if message exists
        if request.message_id not in mail_hierarchy.graph:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": {
                        "code": "RESOURCE_NOT_FOUND",
                        "message": "Message not found",
                        "details": f"Message {request.message_id} does not exist"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Get message info before deletion
        message_info = mail_hierarchy.graph.nodes[request.message_id]
        thread_id = message_info.get("thread_id", "unknown")
        
        # Remove message from graph
        mail_hierarchy.graph.remove_node(request.message_id)
        
        # Remove from embeddings and summaries if they exist
        if request.message_id in mail_hierarchy.node_embeddings:
            del mail_hierarchy.node_embeddings[request.message_id]
        if request.message_id in mail_hierarchy.node_summaries:
            del mail_hierarchy.node_summaries[request.message_id]
        
        # Rebuild vector index if it exists
        if mail_hierarchy.vector_index:
            mail_hierarchy._build_vector_index()
        
        # Save updated graph
        mail_hierarchy.save_graph()
        
        return SuccessResponse(
            message=f"Successfully deleted message {request.message_id}",
            data={
                "message_id": request.message_id,
                "thread_id": thread_id,
                "graph_nodes": len(mail_hierarchy.graph.nodes),
                "graph_edges": len(mail_hierarchy.graph.edges)
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                },
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/maillist/thread/new", response_model=SuccessResponse)
async def create_new_thread(request: NewThreadRequest):
    """
    Create new thread in mail hierarchical graph.
    """
    if main_pipeline is None or rag_system is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "RAG system not initialized",
                    "details": "The mail hierarchy system is not available"
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        # Get mail hierarchical RAG system
        mail_hierarchy = rag_system.graph_manager.mail_hierarchy
        
        # Create thread data structure
        thread_data = {
            "thread_id": request.thread_id,
            "url": request.url or "",
            "subject": request.subject,
            "date_active": request.date_active or "",
            "starting_email": request.starting_email or "",
            "emails_url": request.emails_url or "",
            "replies_count": request.replies_count,
            "votes_total": request.votes_total
        }
        
        # Add thread to hierarchy
        success = mail_hierarchy.add_thread(thread_data)
        
        if success:
            # Save updated graph
            mail_hierarchy.save_graph()
            
            return SuccessResponse(
                message=f"Successfully created thread {request.thread_id}",
                data={
                    "thread_id": request.thread_id,
                    "subject": request.subject,
                    "graph_nodes": len(mail_hierarchy.graph.nodes),
                    "graph_edges": len(mail_hierarchy.graph.edges)
                },
                timestamp=datetime.now().isoformat()
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": {
                        "code": "THREAD_CREATION_FAILED",
                        "message": "Failed to create thread",
                        "details": f"Could not create thread {request.thread_id}"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating new thread: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                },
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/maillist/thread/email", response_model=SuccessResponse)
async def create_email_thread(request: EmailThreadRequest):
    """
    Create new email thread with messages in mail hierarchical graph.
    """
    if main_pipeline is None or rag_system is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "RAG system not initialized",
                    "details": "The mail hierarchy system is not available"
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        # Get mail hierarchical RAG system
        mail_hierarchy = rag_system.graph_manager.mail_hierarchy
        
        # Create thread data structure
        thread_data = {
            "thread_id": request.thread_info.thread_id,
            "url": request.thread_info.url,
            "subject": request.thread_info.subject,
            "date_active": request.thread_info.date_active,
            "starting_email": request.thread_info.starting_email,
            "emails_url": request.thread_info.emails_url,
            "replies_count": request.thread_info.replies_count,
            "votes_total": request.thread_info.votes_total
        }
        
        # Add thread to hierarchy
        thread_success = mail_hierarchy.add_thread(thread_data)
        
        if not thread_success:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": {
                        "code": "THREAD_CREATION_FAILED",
                        "message": "Failed to create thread",
                        "details": f"Could not create thread {request.thread_info.thread_id}"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Add messages to the thread
        added_messages = 0
        failed_messages = []
        
        for message_data in request.messages:
            try:
                message_dict = {
                    "message_id": message_data.message_id,
                    "thread_id": message_data.thread_url,
                    "subject": message_data.subject,
                    "content": message_data.content,
                    "sender_address": message_data.sender_address,
                    "from": message_data.from_field,
                    "date": message_data.date,
                    "to": message_data.to,
                    "cc": message_data.cc,
                    "reply_to": message_data.reply_to,
                    "url": message_data.url,
                    "parent": message_data.parent,
                    "children": message_data.children
                }
                
                success = mail_hierarchy.add_new_message(message_dict)
                if success:
                    added_messages += 1
                else:
                    failed_messages.append(message_data.message_id)
                    
            except Exception as e:
                logger.error(f"Error adding message {message_data.message_id}: {e}")
                failed_messages.append(message_data.message_id)
        
        # Save updated graph
        mail_hierarchy.save_graph()
        
        return SuccessResponse(
            message=f"Successfully created thread with {added_messages} out of {len(request.messages)} messages",
            data={
                "requestId": request.requestId,
                "thread_id": request.thread_info.thread_id,
                "thread_subject": request.thread_info.subject,
                "added_messages": added_messages,
                "total_messages": len(request.messages),
                "failed_messages": failed_messages,
                "graph_nodes": len(mail_hierarchy.graph.nodes),
                "graph_edges": len(mail_hierarchy.graph.edges)
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating email thread: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                },
                "timestamp": datetime.now().isoformat()
            }
        )
