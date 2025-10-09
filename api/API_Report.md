# Boost Knowledge Assistant API Report

## Overview
FastAPI-based REST API for the C++ Boost RAG system. It exposes endpoints to scrape, index, search, query (with optional LLM answer generation), manage chat sessions, and administer an email thread hierarchy used in hybrid retrieval.

- Framework: FastAPI + Pydantic + Loguru
- Outputs: JSON; OpenAPI docs at `/docs`
- UI: Root `/` serves the web interface (config `start_page`)
- Pagination: Supported via `offset` and `limit` in search/query

## Startup and Global State
On startup (`@app.on_event("startup")`):
- Initializes `ImprovedBoostPipeline` → loads configs → creates models, processors, and RAG system.
- Caches available LLMs (from `text_generation.ollama_config`) and embedding models.

Global state:
- `main_pipeline` (ImprovedBoostPipeline)
- `rag_system` (RAGSystem)
- `channel_storage` (in-memory map of chat channels)
- `cached_llm_models`, `cached_embedding_models`

Note: CORS middleware is scaffolded but currently commented out in code. Enable if integrating with external frontends.

## Data Models (Pydantic)
Request models (key fields):
- RagSettingRequest: `embedding`, `database`, `retrieval_weights`, `reranker`, `context_filtering`, `llm`, `evaluation`, `language`
- ScrapeRequest: `source_url`, `max_depth`, `delay`, `max_files`, `use_enhanced_rag`
- QueryRequest: `question`, `offset`, `limit`, plus toggles (`use_chat_history`, `use_multi_step`, `use_evaluation`, `clear_history`) and context (`client_id`, `retrieval_weights`, `embedding`, `database`, `llm`, `language`)
- SearchQueryRequest: `query`, `offset`, `limit`, optional retrieval/embedding/database/language
- EmailThreadRequest: `thread_info` (ThreadInfo) + `messages` (List[MessageData])
- EmailMessagesRequest: `messages` (List[MessageData])
- UpdateMessageRequest: `message_id` and optional fields to update
- DeleteMessageRequest: `message_id`
- NewThreadRequest: thread metadata

Response models (key fields):
- QueryResponse: `answer`, `sources[]`, `retrieval_results[]`, `chat_history`, `embedding_model`, `timestamp`
- SearchResponse: `results[]`, `total_results`, `search_time`, `timestamp`, `offset`, `limit`
- StatusResponse: `pipeline_stats`, `config_loaded`, `components_initialized`, `rag_statistics`
- SuccessResponse / ErrorResponse / ChannelResponse / LLMModelsResponse

## Endpoints
### Core
- GET `/`
  - Serves the configured `start_page` (web UI). Default: `src/templates/index.html`.

- GET `/health`
  - Returns `{ status, timestamp }` if the pipeline is initialized.

- GET `/status` → StatusResponse
  - Aggregated pipeline status, component flags, and RAG statistics.

### Configuration
- POST `/set_pipeline` (RagSettingRequest)
  - Updates RAG runtime properties (LLM group, embedding, language). Returns updated settings.

- GET `/config/rag`
  - Returns: `database_types`, `retrieval_default_weight`, `language_types`, and defaults (`database`, `language`).

### Models
- GET `/models/llm` → LLMModelsResponse
  - Cached list of local LLMs (via Ollama config) with sizes.

- GET `/models/embedding`
  - Returns available embedding models and a default.

- GET `/rag-systems`
  - Metadata describing available RAG systems (currently one: `main`).

### Indexing and Documents
- POST `/index/processed`
  - Indexes `data/source_data/processed/{lang}` by running semantic chunking and loading chunks into RAG.
  - Request query params: `language`, `max_files` (optional).
  - Returns: chunk file list and counts.

- GET `/doc/fulltext?path=...&language=...`
  - Reads full text for a document relative to `data/source_data/processed/{lang}` with path safety checks.

### Search and Query
- POST `/search` (SearchQueryRequest) → SearchResponse
  - Retrieval-only search with hybrid retrieval; supports `offset` and `limit`.

- POST `/query` (QueryRequest) → QueryResponse
  - Runs hybrid retrieval first, then optionally uses an answer generator to produce an LLM answer.
  - Provides both `sources[]` and raw `retrieval_results[]` used to compose the answer.

Pagination behavior:
- `/search`: Request body includes `offset` and `limit`. Response echoes `offset`, `limit`, and `total_results` for that page.
- `/query`: Uses `offset`/`limit` to page retrieval results before generation; pagination fields are not echoed on QueryResponse (see Search for paging metadata).

Examples:
```bash
# Search (offset/limit)
curl -sS -X POST 'http://localhost:8000/search' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Boost.Asio async_read","offset":0,"limit":5}'

# Query with chat history
curl -sS -X POST 'http://localhost:8000/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"How do I manage asynchronous timers in Boost.Asio?",
    "use_chat_history":true,
    "client_id":"demo",
    "offset":0,
    "limit":5
  }'
```

### Scraping and Loading
- POST `/scrape` (ScrapeRequest)
  - Starts a background task to crawl and process pages. Returns an immediate `processing` status.

- POST `/rag/load`
  - Triggers loading of persisted embedding data (when available).

### Chat and Channels
- POST `/chat/clear`
  - Clears global chat history (if available on RAG system).

- GET `/chat/history`
  - Optional query: `include_metadata`, `channel_id`.
  - Returns global or channel-specific history.

- GET `/chat/history/{client_id}`
  - Returns the last `max_messages` messages (default 20) for a specified client.

- DELETE `/chat/history/{client_id}`
  - Clears history for a specified client.

- GET `/chat/session/{client_id}`
  - Returns session statistics for the client.

- GET `/chat/stats`
  - Returns global chat statistics across all sessions.

- POST `/chat/cleanup`
  - Prunes expired chat sessions; returns a count of cleaned sessions.

- POST `/channels` (ChannelRequest) → ChannelResponse
  - Creates a new channel (in-memory).

- GET `/channels` → List[ChannelResponse]
  - Lists channels with message counts and last activity.

- DELETE `/channels/{channel_id}`
  - Deletes a channel.

- POST `/channels/{channel_id}/messages`
  - Appends a message to a channel and updates `last_activity`.

### Email Hierarchy (RAG Mail Graph)
- GET `/mail/message?url=...` or `?message_id=...`
  - Fetches mail metadata and a synthesized content view for a node in the mail graph.

- POST `/maillist/messages/new` (EmailMessagesRequest) → SuccessResponse
  - Adds messages to the mail hierarchical graph (with error tracking per message).

- PUT `/maillist/message/update` (UpdateMessageRequest) → SuccessResponse
  - Updates a message node’s metadata and persists.

- DELETE `/maillist/message/delete` (DeleteMessageRequest) → SuccessResponse
  - Removes a message node and rebuilds indices if needed.

- POST `/maillist/thread/new` (NewThreadRequest) → SuccessResponse
  - Creates a new thread node in the graph.

- POST `/maillist/thread/email` (EmailThreadRequest) → SuccessResponse
  - Creates a thread and adds messages in one call; persists the graph.

## Errors and Status Codes
- 200: success
- 400: validation or operation failure
- 404: resource not found
- 500: internal error
- 503: service unavailable (pipeline/RAG not initialized)

Error body shape:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details"
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## Security and CORS
- Authentication/authorization: not implemented (deploy behind a trusted gateway or add auth middleware).
- CORS middleware scaffolding present but disabled; enable and configure if serving cross-origin clients.

## Operational Notes
- Root UI path is configurable: `start_page` in `config.yaml`.
- Indexing via `/index/processed` is non-destructive and loads data into vector/BM25/graph stores.
- Query runs hybrid retrieval then (if available) uses the configured answer generator; when not available, it falls back to concatenated snippets.
- Pagination is consistent through `offset`/`limit` across search/query requests.

## Quick Start
```bash
# Start the API
python src/api/run.py

# Health check
curl -sS http://localhost:8000/health

# Search
curl -sS -X POST http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"asio timer","offset":0,"limit":5}'

# Query
curl -sS -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question":"What is Boost.Asio?","offset":0,"limit":5}'
```

## Detailed Field Reference (Pydantic BaseModels)

### RagSettingRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| embedding | str | No | None | Embedding type identifier |
| database | str | No | None | Vector DB type |
| retrieval_weights | Dict[str, float] | No | None | Weights per retrieval method |
| reranker | str | No | None | Reranker type |
| context_filtering | str | No | None | Context filtering type |
| llm | str | No | None | LLM provider/type |
| evaluation | str | No | None | Evaluation type |
| language | str | No | None | RAG language |

### ScrapeRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| source_url | str | Yes | - | URL to scrape |
| max_depth | int | No | 2 | Max crawl depth |
| delay | float | No | 1.0 | Delay between requests (s) |
| max_files | int | No | None | Max files to process |
| use_enhanced_rag | bool | No | False | Use enhanced RAG (flag) |

### QueryRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| question | str | Yes | - | User question |
| offset | int | No | 0 | Retrieval offset |
| limit | int | No | 5 | Retrieval page size |
| use_enhanced_rag | bool | No | False | Use enhanced RAG features |
| use_multi_step | bool | No | False | Enable multi-step reasoning |
| use_evaluation | bool | No | False | Enable evaluation mode |
| use_chat_history | bool | No | True | Include chat history in context |
| clear_history | bool | No | False | Clear history before processing |
| client_id | str | No | "default_client" | Client identifier for chat |
| retrieval_weights | Dict[str, float] | No | None | Weights per retrieval method |
| embedding | str | No | None | Embedding model override |
| database | str | No | None | Vector DB override |
| llm | str | No | None | LLM override |
| language | str | No | None | Language override |

### SearchQueryRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| query | str | Yes | - | Search query |
| offset | int | No | 0 | Retrieval offset |
| limit | int | No | 10 | Retrieval page size |
| retrieval_weights | Dict[str, float] | No | None | Weights per retrieval method |
| embedding | str | No | None | Embedding model override |
| database | str | No | None | Vector DB override |
| language | str | No | None | Language override |

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

### EmailMessagesRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| timestamp | str | Yes | - | Request timestamp |
| requestId | str | Yes | - | Unique request ID |
| messages | List[MessageData] | Yes | - | Messages array |
| message_count | int | Yes | - | Number of messages |

### UpdateMessageRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| message_id | str | Yes | - | Message ID to update |
| subject | str | No | None | Updated subject |
| content | str | No | None | Updated content |
| sender_address | str | No | None | Updated sender |
| from_field | str | No | None | Updated from |
| date | str | No | None | Updated date |
| to | str | No | None | Updated to |
| cc | str | No | None | Updated CC |
| reply_to | str | No | None | Updated Reply-To |
| url | str | No | None | Updated URL |

### DeleteMessageRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| message_id | str | Yes | - | Message ID to delete |

### NewThreadRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| thread_id | str | Yes | - | Unique thread ID |
| subject | str | Yes | - | Thread subject |
| url | str | No | None | Thread URL |
| date_active | str | No | None | Active date |
| starting_email | str | No | None | Starting email ID |
| emails_url | str | No | None | Emails list URL |
| replies_count | int | No | 0 | Replies count |
| votes_total | int | No | 0 | Votes total |

### QueryResponse
| Field | Type | Description |
|---|---|---|
| question | str | Original question |
| answer | str | Generated/fallback answer |
| sources | List[Dict[str, Any]] | Final sources used in the answer |
| retrieval_results | List[Dict[str, Any]] | Raw retrieval items (optional) |
| timestamp | str | ISO timestamp |
| chat_history | List[Dict[str, Any]] | Recent history (optional) |
| history_length | int | Number of entries in history |
| embedding_model | str | Embedding model used (optional) |

### SearchResponse
| Field | Type | Description |
|---|---|---|
| query | str | Search query |
| results | List[Dict[str, Any]] | Retrieval results (paged) |
| total_results | int | Count in current page |
| search_time | float | Seconds elapsed |
| timestamp | str | ISO timestamp |
| offset | int | Start offset used |
| limit | int | Page size used |

### StatusResponse
| Field | Type | Description |
|---|---|---|
| pipeline_stats | Dict[str, Any] | Pipeline statistics |
| config_loaded | bool | Config load status |
| components_initialized | Dict[str, bool] | Component init map |
| rag_statistics | Dict[str, Any] | RAG stats (optional) |

### ModelInfo
| Field | Type | Description |
|---|---|---|
| name | str | Model name |
| size | int | Model size (bytes) |
| parameter_size | str | Human-readable params size |

### LLMModelsResponse
| Field | Type | Description |
|---|---|---|
| models | List[ModelInfo] | Available LLM models |
| total_count | int | Number of models |
| sorted_by | str | Sort descriptor |

### EvaluationRequest
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| dataset_path | str | No | data/validation_dataset.json | Dataset path |
| max_questions | int | No | 5 | Number of questions to evaluate |

### EvaluationResponse
| Field | Type | Description |
|---|---|---|
| total_questions | int | Questions evaluated |
| average_similarity | float | Average similarity |
| average_time | float | Average time per question |
| accuracy | float | Accuracy metric |
| good_answers | int | Count of good answers |
| results | List[Dict[str, Any]] | Per-question details |

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

### ActionStatsResponse
| Field | Type | Description |
|---|---|---|
| success | bool | Operation success flag |
| message | str | Human-readable message |
| data | Dict[str, Any] | Extra data |
| timestamp | str | ISO timestamp |
