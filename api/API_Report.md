# VectorDataBuild API Report

## 📋 Overview

The VectorDataBuild API is a comprehensive REST API built with FastAPI that provides endpoints for scraping, processing, and querying vector data with dual RAG (Retrieval-Augmented Generation) support. The API serves as the main interface for the C++ Boost RAG system.

## 🏗️ Architecture

### Core Components
- **FastAPI Framework**: Modern, fast web framework for building APIs
- **Pydantic Models**: Type-safe request/response validation
- **Background Tasks**: Asynchronous processing for long-running operations
- **CORS Support**: Cross-origin resource sharing
- **Logging Integration**: Comprehensive logging with Loguru

### Global State Management
- **Pipeline Orchestrator**: Main pipeline for data processing
- **RAG System**: Retrieval-Augmented Generation system
- **Chat History Manager**: Multi-client conversation management
- **Channel Storage**: In-memory channel management
- **Model Caching**: Cached LLM and embedding models

## 📡 API Endpoints

### 🏠 **Core Endpoints**

#### `GET /`
- **Purpose**: Root endpoint with API information
- **Response**: Serves the main web interface (index.html)
- **Status**: ✅ Active

#### `GET /health`
- **Purpose**: Health check endpoint
- **Response**: System status and timestamp
- **Status**: ✅ Active

#### `GET /status`
- **Purpose**: Get comprehensive pipeline status
- **Response**: Pipeline stats, component status, RAG statistics
- **Status**: ✅ Active

### 🔧 **Configuration Endpoints**

#### `POST /set_pipeline`
- **Purpose**: Configure RAG pipeline settings
- **Request**: RagSettingRequest (embedding, database, retrieval weights, etc.)
- **Response**: Configuration update confirmation
- **Status**: ✅ Active

#### `GET /config/rag`
- **Purpose**: Get RAG configuration options
- **Response**: Available database types, retrieval weights, language options
- **Status**: ✅ Active

### 🤖 **Model Management**

#### `GET /models/llm`
- **Purpose**: Get available LLM models
- **Response**: List of models with size and parameter information
- **Status**: ✅ Active

#### `GET /models/embedding`
- **Purpose**: Get available embedding models
- **Response**: List of embedding models with default selection
- **Status**: ✅ Active

#### `GET /rag-systems`
- **Purpose**: Get information about available RAG systems
- **Response**: System descriptions, features, and initialization status
- **Status**: ✅ Active

### 📊 **Data Processing**

#### `POST /scrape`
- **Purpose**: Scrape data from websites
- **Request**: ScrapeRequest (URL, depth, delay, max files)
- **Response**: Scraping initiation confirmation
- **Background Processing**: ✅ Yes
- **Status**: ✅ Active

#### `POST /rag/load`
- **Purpose**: Load RAG system with embedding data
- **Response**: Loading confirmation
- **Status**: ✅ Active

### 🔍 **Search & Query**

#### `POST /query`
- **Purpose**: Query the RAG system with questions
- **Request**: QueryRequest (question, max results, options)
- **Response**: QueryResponse (answer, sources, chat history)
- **Features**: 
  - Multi-step reasoning
  - Chat history integration
  - Context filtering
  - Source attribution
- **Status**: ✅ Active

#### `POST /search`
- **Purpose**: Search for relevant documents (retrieval only)
- **Request**: SearchQueryRequest (query, max results, options)
- **Response**: SearchResponse (results, search time)
- **Status**: ✅ Active

#### `GET /rag/stats`
- **Purpose**: Get RAG system statistics
- **Response**: System performance metrics
- **Status**: ✅ Active

### 💬 **Chat Management**

#### `POST /chat/clear`
- **Purpose**: Clear global chat history
- **Response**: Confirmation message
- **Status**: ✅ Active

#### `GET /chat/history`
- **Purpose**: Get chat history (global or channel-specific)
- **Parameters**: include_metadata, channel_id
- **Response**: Chat history with metadata
- **Status**: ✅ Active

#### `GET /chat/history/{client_id}`
- **Purpose**: Get chat history for specific client
- **Parameters**: max_messages, include_metadata
- **Response**: Client-specific chat history
- **Status**: ✅ Active

#### `DELETE /chat/history/{client_id}`
- **Purpose**: Clear chat history for specific client
- **Response**: Confirmation message
- **Status**: ✅ Active

#### `GET /chat/session/{client_id}`
- **Purpose**: Get session statistics for client
- **Response**: Session metrics and statistics
- **Status**: ✅ Active

#### `GET /chat/stats`
- **Purpose**: Get global chat statistics
- **Response**: System-wide chat metrics
- **Status**: ✅ Active

#### `POST /chat/cleanup`
- **Purpose**: Clean up expired chat sessions
- **Response**: Cleanup statistics
- **Status**: ✅ Active

### 📧 **Email Thread Management**

#### `POST /maillist/messages/new`
- **Purpose**: Add new messages to mail hierarchical RAG
- **Request**: EmailMessagesRequest (messages array)
- **Response**: Processing results
- **Status**: ✅ Active

#### `PUT /maillist/message/update`
- **Purpose**: Update existing message in mail hierarchy
- **Request**: UpdateMessageRequest (message_id, updated fields)
- **Response**: Update confirmation
- **Status**: ✅ Active

#### `DELETE /maillist/message/delete`
- **Purpose**: Delete message from mail hierarchy
- **Request**: DeleteMessageRequest (message_id)
- **Response**: Deletion confirmation
- **Status**: ✅ Active

#### `POST /maillist/thread/new`
- **Purpose**: Create new thread in mail hierarchy
- **Request**: NewThreadRequest (thread information)
- **Response**: Thread creation confirmation
- **Status**: ✅ Active

#### `POST /maillist/thread/email`
- **Purpose**: Create email thread with messages
- **Request**: EmailThreadRequest (thread info + messages)
- **Response**: Thread and message creation confirmation
- **Status**: ✅ Active

### 🏢 **Channel Management**

#### `POST /channels`
- **Purpose**: Create new chat channel
- **Request**: ChannelRequest (channel_id, channel_name)
- **Response**: ChannelResponse (channel details)
- **Status**: ✅ Active

#### `GET /channels`
- **Purpose**: List all available chat channels
- **Response**: List of channels with statistics
- **Status**: ✅ Active

#### `DELETE /channels/{channel_id}`
- **Purpose**: Delete chat channel
- **Response**: Deletion confirmation
- **Status**: ✅ Active

#### `POST /channels/{channel_id}/messages`
- **Purpose**: Add message to specific channel
- **Request**: Message data (type, text, timestamp)
- **Response**: Message addition confirmation
- **Status**: ✅ Active

## 📋 **Data Models**

### Request Models
- **RagSettingRequest**: RAG configuration settings
- **ScrapeRequest**: Web scraping parameters
- **QueryRequest**: Question-answering parameters
- **SearchQueryRequest**: Document search parameters
- **EmailThreadRequest**: Email thread creation
- **EmailMessagesRequest**: Email message processing
- **UpdateMessageRequest**: Message update operations
- **DeleteMessageRequest**: Message deletion
- **NewThreadRequest**: Thread creation
- **ChannelRequest**: Channel management

### Response Models
- **QueryResponse**: Query results with sources
- **SearchResponse**: Search results with metadata
- **StatusResponse**: System status information
- **SuccessResponse**: Operation success confirmation
- **ErrorResponse**: Error details and codes
- **ChannelResponse**: Channel information
- **LLMModelsResponse**: Available LLM models
- **ActionStatsResponse**: Action statistics

## 🔒 **Error Handling**

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request (validation errors)
- **404**: Not Found (resource not found)
- **500**: Internal Server Error
- **503**: Service Unavailable (pipeline not initialized)

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🚀 **Performance Features**

### Background Processing
- **Scraping Operations**: Asynchronous web scraping
- **Data Processing**: Non-blocking data ingestion
- **Model Loading**: Cached model initialization

### Caching Strategy
- **LLM Models**: Cached at startup for fast access
- **Embedding Models**: Pre-loaded and cached
- **Channel Data**: In-memory channel storage
- **Graph Data**: Persistent graph storage

### Optimization Features
- **Model Caching**: Pre-loaded models for faster responses
- **Background Tasks**: Non-blocking operations
- **Connection Pooling**: Efficient database connections
- **Memory Management**: Optimized data structures

## 📊 **Monitoring & Logging**

### Logging Configuration
- **API Logs**: `logs/api.log` with rotation
- **Log Level**: INFO with structured logging
- **Loguru Integration**: Advanced logging features

### Metrics Available
- **System Status**: Pipeline and component health
- **RAG Statistics**: Retrieval and generation metrics
- **Chat Statistics**: Conversation metrics
- **Performance Metrics**: Response times and throughput

## 🔧 **Configuration**

### API Settings
- **Host**: Configurable (default: 0.0.0.0)
- **Port**: Configurable (default: 8000)
- **Debug Mode**: Development vs production
- **Workers**: Multi-process support

### CORS Configuration
- **Origins**: Configurable allowed origins
- **Methods**: All HTTP methods supported
- **Headers**: All headers allowed
- **Credentials**: Credential support enabled

## 📈 **Usage Statistics**

### Endpoint Categories
- **Core Endpoints**: 3 endpoints
- **Configuration**: 2 endpoints
- **Model Management**: 3 endpoints
- **Data Processing**: 2 endpoints
- **Search & Query**: 3 endpoints
- **Chat Management**: 7 endpoints
- **Email Management**: 5 endpoints
- **Channel Management**: 4 endpoints

### Total Endpoints: 29

## 🎯 **Key Features**

### Advanced RAG Capabilities
- **Hybrid Retrieval**: Vector + BM25 + Graph + Hierarchical search
- **Multi-step Reasoning**: Complex query decomposition
- **Context Filtering**: Relevance and redundancy filtering
- **Source Attribution**: Detailed source tracking
- **Chat History**: Persistent conversation context

### Email Thread Processing
- **Hierarchical Structure**: Thread and message relationships
- **Graph-based Storage**: NetworkX graph representation
- **Message Management**: CRUD operations for messages
- **Thread Organization**: Structured email thread handling

### Multi-client Support
- **Channel Management**: Multiple conversation channels
- **Client Isolation**: Separate chat histories per client
- **Session Management**: Automatic session cleanup
- **Global Statistics**: System-wide metrics

## 🔮 **Future Enhancements**

### Planned Features
- **Real-time WebSocket Support**: Live chat capabilities
- **Advanced Analytics**: Detailed usage metrics
- **Rate Limiting**: API usage controls
- **Authentication**: User management system
- **API Versioning**: Backward compatibility support

### Performance Improvements
- **Database Integration**: Persistent storage for channels
- **Caching Layer**: Redis integration
- **Load Balancing**: Multi-instance support
- **Monitoring Dashboard**: Real-time system monitoring

## 📝 **Conclusion**

The VectorDataBuild API provides a comprehensive, production-ready interface for the C++ Boost RAG system. With 29 endpoints covering data processing, querying, chat management, and email thread processing, it offers a complete solution for knowledge management and question-answering systems.

The API demonstrates excellent architecture with proper error handling, background processing, multi-client support, and advanced RAG capabilities. It's well-suited for both development and production environments with comprehensive logging, monitoring, and configuration options.
