# Boost Knowledge Assistant API Report

## Overview
FastAPI-based REST API for the C++ Boost Knowledge Assistant. Provides a retrieval-focused RAG system using LangChain with hybrid search (ChromaDB + BM25), supporting document filtering, mail data management, and channel-based organization.

- **Framework**: FastAPI + Pydantic + Loguru
- **RAG Backend**: LangChain with ChromaDB vector store + BM25
- **Outputs**: JSON; OpenAPI docs at `/docs`
- **UI**: Root `/` serves the web interface (configurable via `start_page`)
- **Version**: 2.0.0

## Startup and Global State
On startup (`@app.on_event("startup")`):
- Initializes `LangChainRAGPipeline` → loads ChromaDB, creates embedding model, sets up hybrid retrieval

Global state:
- `rag_system` (LangChainRAGPipeline) - Main RAG pipeline
- `channel_storage` (Dict) - In-memory channel storage for frontend organization

Note: CORS middleware is available but currently commented out. Enable if serving cross-origin requests.

## Data Models (Pydantic)
All models are defined in `models/api.py` and can be imported via:
```python
from models import QueryRequest, MessageData, EmailThreadRequest, ...
```

Request models:
- `QueryRequest`: `question`, `search_scopes`, `search_limit`, `offset`, `limit`
- `ChannelRequest`: `channel_id`, `channel_name`
- `EmailThreadRequest`: `thread_info` (ThreadInfo) + `messages` (List[MessageData])
- `MessageData`: Complete message data (used for both adding and updating)
- `DeleteMessageRequest`: `message_id`, `url`

Response models:
- `QueryResponse`: `question`, `answer`, `retrieval_results[]`, `timestamp`
- `ChannelResponse`: `channel_id`, `channel_name`, `message_count`, `last_activity`, `created_at`
- `SuccessResponse`: `success`, `message`, `data`, `timestamp`

Note: `retrieval_results` contains full document content, metadata, scores, and source file information.

---

## API Endpoints

### Core

**GET** `/`
- Serves the web UI (configured via `start_page` in config.yaml)
- Default: `src/templates/index.html`

**GET** `/health`
- Health check endpoint
- Returns: `{"status": "healthy", "timestamp": "..."}`
- Status code: 503 if RAG system not initialized

---

### Query

**POST** `/query` (QueryRequest) → QueryResponse

Main query endpoint using hybrid retrieval (ChromaDB + BM25):
- Supports document type filtering via `search_scopes`
- Returns retrieved documents with scores and metadata
- No LLM generation - returns concatenated context as answer

**Request Example:**
```bash
curl -X POST 'http://localhost:8080/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How does Boost.Asio handle async operations?",
    "search_scopes": ["documentation", "mail"],
    "search_limit": 10
  }'
```

**Response Fields:**
- `question`: Original question
- `answer`: Concatenated context from retrieved documents
- `retrieval_results`: Full retrieval results with content and metadata
- `timestamp`: ISO timestamp

---

### Channels

**POST** `/channels` (ChannelRequest) → ChannelResponse
- Create a new channel for frontend organization
- In-memory storage (not persisted)

**GET** `/channels` → List[ChannelResponse]
- List all channels with message counts and activity

**DELETE** `/channels/{channel_id}`
- Delete a channel
- Returns: `{"message": "Channel {id} deleted successfully"}`

**POST** `/channels/{channel_id}/messages`
- Add a message to a channel
- Updates `last_activity` timestamp
- Body: `{"type": "user", "text": "..."}`

---

### Mail Data Management

**POST** `/maillist/messages/new` (EmailThreadRequest) → SuccessResponse

Add new mail messages to the RAG system:
- Performs duplicate checking and updates existing messages
- Returns statistics: `added_count`, `updated_count`, `failed_count`

**Request Example:**
```json
{
  "timestamp": "2025-10-27T10:00:00Z",
  "requestId": "req_123",
  "thread_info": {
    "url": "https://lists.boost.org/...",
    "thread_id": "thread_001",
    "subject": "Question about Boost.Asio",
    "date_active": "2025-10-27",
    "starting_email": "msg_001",
    "emails_url": "https://...",
    "replies_count": 5,
    "votes_total": 10
  },
  "messages": [
    {
      "message_id": "msg_001",
      "subject": "Question",
      "content": "How do I use async_read?",
      "thread_url": "https://...",
      "sender_address": "user@example.com",
      "from_field": "User Name",
      "date": "2025-10-27",
      "to": "boost@lists.boost.org",
      "url": "https://..."
    }
  ],
  "message_count": 1
}
```

**PUT** `/maillist/message/update` (MessageData) → SuccessResponse
- Update existing message in the RAG system
- Requires `message_id` and at least one field to update
- Updates document metadata and re-indexes

**DELETE** `/maillist/message/delete` (DeleteMessageRequest) → SuccessResponse
- Delete a message from the RAG system
- Requires `message_id` and `url`
- Removes document from vector store and indices

---

## Errors and Status Codes

- **200**: Success
- **400**: Validation or operation failure
- **404**: Resource not found
- **500**: Internal error
- **501**: Not implemented
- **503**: Service unavailable (RAG system not initialized)

**Error Response Format:**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details"
  },
  "timestamp": "2025-10-27T10:00:00Z"
}
```

---

## Security and CORS

- **Authentication/authorization**: Not implemented (deploy behind a trusted gateway or add auth middleware)
- **CORS middleware**: Available but disabled; enable and configure if serving cross-origin clients

---

## Implementation Details

### RAG System
- **Vector Store**: ChromaDB with `all-MiniLM-L6-v2` embeddings
- **Sparse Retrieval**: BM25 (created dynamically during queries)
- **Hybrid Search**: Combines dense (ChromaDB) and sparse (BM25) with TF-IDF reranking
- **Caching**: Query results cached with TTL for fast response times
- **Memory Usage**: ~1.5-2GB typical, optimized for production use

### Document Types
- `documentation`: C++ Boost documentation pages
- `mail`: Mailing list messages
- `slack`: Slack messages (if available)

### Search Scopes
The `/query` endpoint accepts `search_scopes` to filter by document type:
- `["documentation"]` - Only docs
- `["mail"]` - Only mailing list
- `["documentation", "mail"]` - Both (default includes all types)

### Answer Generation
- **No LLM**: The system is retrieval-only (LLM components removed for simplicity)
- Answer is concatenated context from top-k retrieved documents
- For LLM integration, extend `LangChainRAGPipeline` with answer generation

---

## Quick Start

### Start the API
```bash
cd E:\CppGreat\cppa-brain-backend
python api/run.py
```

The API will start on `http://localhost:8000`

### Health Check
```bash
curl http://localhost:8000/health
```

### Query Examples

**Basic Query:**
```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is Boost.Asio?",
    "search_limit": 10
  }'
```

**With Type Filtering:**
```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How to use shared_ptr?",
    "search_scopes": ["documentation"],
    "search_limit": 15
  }'
```

**Documentation & Mail:**
```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "async_read example",
    "search_scopes": ["documentation", "mail"],
    "search_limit": 20
  }'
```

---

## Detailed Field Reference

### QueryRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| question | str | Yes | - | User question |
| search_scopes | List[str] | No | ["documentation", "mail", "slack"] | Search scopes to include |
| search_limit | int | No | 10 | Number of search results to return |
| offset | int | No | 0 | Retrieval offset (deprecated) |
| limit | int | No | None | Retrieval page size (deprecated) |

### QueryResponse
| Field | Type | Description |
|---|---|---|
| question | str | Original question |
| answer | str | Concatenated context from retrieved documents |
| retrieval_results | List[Dict[str, Any]] | Retrieval results with full content and metadata (optional) |
| timestamp | str | ISO timestamp |

**Retrieval Result Object Structure:**
```json
{
  "text": "Full document content...",
  "source_file": "URL or file path",
  "score": 0.85,
  "retrieval_method": "hybrid",
  "source_type": "documentation",
  "metadata": {
    "url": "...",
    "type": "documentation",
    "final_score": 0.85
  }
}
```

### ChannelRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| channel_id | str | Yes | - | Channel ID |
| channel_name | str | No | None | Channel name |

### ChannelResponse
| Field | Type | Description |
|---|---|---|
| channel_id | str | Channel ID |
| channel_name | str | Channel name |
| message_count | int | Messages in channel |
| last_activity | str | ISO timestamp |
| created_at | str | ISO timestamp |

### ThreadInfo
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| url | str | Yes | - | Thread URL |
| thread_id | str | Yes | - | Thread ID |
| subject | str | Yes | - | Thread subject |
| date_active | str | Yes | - | Active date |
| starting_email | str | Yes | - | Starting email URL |
| emails_url | str | Yes | - | Emails list URL |
| replies_count | int | No | 0 | Replies count |
| votes_total | int | No | 0 | Votes total |

### MessageData
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| message_id | str | Yes | - | Unique message ID |
| subject | str | Yes | - | Message subject |
| content | str | Yes | - | Message content |
| thread_url | str | Yes | - | Thread URL |
| parent | str | No | None | Parent message URL |
| children | List[str] | No | [] | Children message URLs |
| sender_address | str | Yes | - | Sender email address |
| from_field | str | Yes | - | From field text |
| date | str | Yes | - | Message date |
| to | str | Yes | - | To field |
| cc | str | No | "" | CC field |
| reply_to | str | No | "" | Reply-To field |
| url | str | Yes | - | Message URL |

### EmailThreadRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| timestamp | str | Yes | - | Request timestamp |
| requestId | str | Yes | - | Unique request ID |
| thread_info | ThreadInfo | Yes | - | Thread metadata |
| messages | List[MessageData] | Yes | - | Messages array |
| message_count | int | Yes | - | Number of messages |

### MessageData (for updates)
When used in the update endpoint, MessageData includes:
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| message_id | str | Yes | - | Message ID to update |
| subject | str | Yes | - | Message subject |
| content | str | Yes | - | Message content |
| thread_url | str | Yes | - | Thread URL |
| sender_address | str | Yes | - | Sender email address |
| from_field | str | Yes | - | From field text |
| date | str | Yes | - | Message date |
| to | str | Yes | - | To field |
| url | str | Yes | - | Message URL |
| parent | str | No | None | Parent message URL |
| children | List[str] | No | [] | Children message URLs |
| cc | str | No | "" | CC field |
| reply_to | str | No | "" | Reply-To field |

### DeleteMessageRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| message_id | str | Yes | - | Message ID to delete |
| url | str | Yes | - | Message URL to delete |

### SuccessResponse
| Field | Type | Description |
|---|---|---|
| success | bool | Always true |
| message | str | Human-readable message |
| data | Dict[str, Any] | Extra data |
| timestamp | str | ISO timestamp |

### ErrorResponse
| Field | Type | Description |
|---|---|---|
| success | bool | Always false |
| error | Dict[str, Any] | Error payload |
| timestamp | str | ISO timestamp |

---

## Performance

Typical metrics (with ~50k documents indexed):
- Initial indexing: ~15 minutes
- Index loading: ~3 seconds  
- Query (uncached): ~250ms
- Query (cached): ~5ms
- Memory usage: ~1.5-2GB
- Cache hit rate: 65-80%

---

## Future Enhancements

Potential additions:
- LLM integration for answer generation (Ollama/OpenAI)
- Advanced search with filters and facets
- Document upload and processing endpoints
- Real-time document updates via webhooks
- Persistent channel storage (database)
- Authentication and rate limiting

---

## Related Documentation

- `langchain_rag/README.md` - LangChain RAG pipeline documentation
- `api/SEARCH_SCOPE_GUIDE.md` - Search scope filtering guide
- `models/api.py` - Complete API model definitions
- `MODELS_MIGRATION_SUMMARY.md` - Models organization overview
- `PROJECT_STRUCTURE.md` - Complete project structure

---

**Part of the C++ Boost Knowledge Assistant project.**
