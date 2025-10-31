"""
FastAPI-based REST API for C++ Boost Assistant.
Provides endpoints for scraping, processing, querying, and managing the knowledge base.
"""

import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import LangChain RAG pipeline directly
from langchain_rag.rag_pipeline import LangChainRAGPipeline
from config import get_config

# Import API models
from models import (
    QueryRequest,
    ChannelRequest,
    QueryResponse,
    ChannelResponse,
    SuccessResponse,
    MessageData,
    EmailThreadRequest,
    DeleteMessageRequest,
)


# Search method/scope enums for compatibility
class SearchMethod:
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    GRAPH = "graph"


class SearchScope:
    DOCS = "docs"
    MAIL = "mail"
    BOTH = "both"


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
    version="2.0.0",
)

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Global pipeline instance
rag_system: Optional[LangChainRAGPipeline] = None

# Simple in-memory channel storage (in production, use a proper database)
channel_storage: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the LangChain RAG pipeline on startup."""
    global rag_system
    try:
        # Create LangChain RAG pipeline directly
        rag_system = LangChainRAGPipeline()

        logger.info("Boost Knowledge Assistant API started successfully with LangChain RAG support")
    except Exception as e:
        logger.error(f"Failed to initialize LangChain RAG pipeline: {e}")
        raise


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
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


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
            "last_activity": datetime.now().isoformat(),
        }

        return ChannelResponse(
            channel_id=channel_id,
            channel_name=channel_name,
            message_count=0,
            last_activity=datetime.now().isoformat(),
            created_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error creating channel: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/channels", response_model=List[ChannelResponse])
async def list_channels():
    """
    List all available chat channels.
    """
    try:
        channels = []
        for channel_id, channel_data in channel_storage.items():
            channels.append(
                ChannelResponse(
                    channel_id=channel_id,
                    channel_name=channel_data.get("name", "Unknown"),
                    message_count=len(channel_data.get("messages", [])),
                    last_activity=channel_data.get("last_activity", ""),
                    created_at=channel_data.get("created_at", ""),
                )
            )

        # Sort by last activity
        channels.sort(key=lambda x: x.last_activity, reverse=True)
        return channels
    except Exception as e:
        logger.error(f"Error listing channels: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
            "timestamp": datetime.now().isoformat(),
        }

        channel["messages"].append(message_data)
        channel["last_activity"] = datetime.now().isoformat()

        return {"message": "Message added successfully"}
    except Exception as e:
        logger.error(f"Error adding message to channel: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Supports filtering by search scope (documentation, mail, slack) and customizable search limit.
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Use search_limit as primary, fall back to limit + offset for backward compatibility
        fetch_k = request.search_limit
        if request.limit is not None and request.limit > 0:
            fetch_k = request.offset + request.limit
        
        # Normalize search scopes
        filter_types = None
        if request.search_scopes and len(request.search_scopes) > 0:
            filter_types = request.search_scopes
        
        logger.info(
            f"Query: '{request.question}' | fetch_k={fetch_k} | filter_types={filter_types}"
        )

        # Query LangChain RAG system with filtering
        rag_response = rag_system.query(
            question=request.question,
            fetch_k=fetch_k,
            filter_types=filter_types
        )

        # Extract response data
        answer = rag_response.get("answer", "")
        source_documents = rag_response.get("source_documents", [])
        
        # Build retrieval_results from Document objects
        # Note: Results are already limited by fetch_k in the query
        retrieval_results = []
        for i, doc in enumerate(source_documents):
            if i < request.offset:
                continue
            # Get document type for display
            doc_type = doc.metadata.get("type", "documentation")
            
            result = {
                "text": doc.page_content,
                "source_file": doc.metadata.get("url", doc.metadata.get("source", "unknown")),
                "score": doc.metadata.get("final_score", 1.0 / (i + 1)),
                "retrieval_method": "hybrid",
                "source_type": doc_type,
                "metadata": doc.metadata
            }
            retrieval_results.append(result)

        # Log retrieval results for debugging
        logger.info(
            f"📊 Retrieved {len(retrieval_results)} results for query: '{request.question}'"
        )

        # Count retrieval methods
        method_counts = {}
        for i, result in enumerate(retrieval_results):
            method = result.get("retrieval_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
            logger.info(
                f"   Result {i+1}: {result.get('source_file', 'Unknown')} (score: {result.get('score', 0):.3f}, method: {method})"
            )

        # Log retrieval method summary
        if method_counts:
            logger.info("📈 Retrieval method breakdown:")
            for method, count in method_counts.items():
                logger.info(f"   {method}: {count} results")

        return QueryResponse(
            question=request.question,
            answer=answer,
            retrieval_results=retrieval_results,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.put("/maillist/message/update", response_model=SuccessResponse)
async def update_message(request: MessageData):
    """
    Update existing message in mail hierarchical graph.
    Note: Mail hierarchy not supported in LangChain RAG version.
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    try:
        success = rag_system.update_mail_data(request)
        if success:
            return SuccessResponse(
                success=True,
                message=f"Message {request.message_id} updated successfully",
                data={
                    "message_id": request.message_id,
                },
                timestamp=datetime.now().isoformat()
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to update message")
    except Exception as e:
        logger.error(f"Error updating message: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    

@app.delete("/maillist/message/delete", response_model=SuccessResponse)
async def delete_message(request: DeleteMessageRequest):
    """
    Delete message from mail hierarchical graph.
    Note: Mail hierarchy not supported in LangChain RAG version.
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    try:
        success = rag_system.delete_document(request.url)
        if success:
            return SuccessResponse(
                success=success,
                message=f"Message {request.message_id} deleted successfully",
                data={"message_id": request.message_id},
                timestamp=datetime.now().isoformat()
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to delete message")
    except Exception as e:
        logger.error(f"Error deleting message: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    
@app.post("/maillist/messages/new", response_model=SuccessResponse)
async def add_new_messages(request: EmailThreadRequest):
    """
    Add new messages to mail RAG database.
    """
    if rag_system is None:
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
        # Add mail data with duplicate checking
        result = rag_system.add_mail_data(request.messages)
        
        # Extract statistics
        added_count = result.get("added_count", 0)
        updated_count = result.get("updated_count", 0)
        failed_count = result.get("failed_count", 0)
        failed_messages = result.get("failed_messages", [])
        total_processed = result.get("total_processed", len(request.messages))
        
        # Create response message
        message = (
            f"Processed {total_processed} messages: {added_count} added, "
            f"{updated_count} updated (already exist), "
            f"{failed_count} failed"
        )
        return SuccessResponse(
            message=message,
            data={
                "requestId": request.requestId,
                "thread_id": request.thread_info.thread_id,
                "thread_subject": request.thread_info.subject,
                "added_messages": added_count,
                "updated_messages": updated_count,
                "total_messages": total_processed,
                "failed_messages": failed_messages,
                "graph_nodes": 0,
                "graph_edges": 0
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
        ) from e

