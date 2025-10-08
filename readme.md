## C++ Boost RAG System

An end-to-end Retrieval-Augmented Generation (RAG) stack for the C++ Boost ecosystem (docs and mailing lists). It combines robust data processing, hybrid retrieval, reranking, optional multi-step reasoning, and a FastAPI server for querying.

### Key features
- **Multi-format processing**: HTML/Markdown/JSON and email archives
- **Semantic chunking** with adaptive windowing
- **Hybrid retrieval**: vectors (FAISS/Chroma) + BM25 + graphs + hierarchical mail search
- **Reranking**: cross-encoder reranker
- **LLMs**: Ollama, OpenAI, Gemini, HuggingFace
- **API**: FastAPI app with docs at `/docs`

## Project structure
```
boost_rag_project/
├── config/
│   └── config.yaml
├── data/
├── requirements.txt
├── src/
│   ├── api/                 # FastAPI app (`run.py`)
│   ├── data_processor/      # Parsing, semantic chunking, summarization
│   ├── rag/                 # Vector/BM25/graph/mail RAG
│   ├── templates/           # Web UI (landing page)
│   ├── text_generation/     # LLM adapters and answer generator
│   ├── utils/               # Config helpers
│   └── main_pipeline.py     # Orchestrator and CLI entry
└── README.md
```

## Installation
Prerequisites: Python 3.10+ recommended, Git. CUDA is optional; CPU works.

```bash
git clone <repo-url>
cd boost_rag_project
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration
Primary config: `config/config.yaml`.
- **Embeddings**: `rag.embedding.*` (default `minilm`)
- **Vector DB**: `rag.database.{faiss|chroma}` (default `faiss`)
- **Search**: `rag.retrieval.search_engine.{bm25|elasticsearch}` (default `bm25`)
- **Graphs**: `rag.retrieval.graph.backend` (`networkx` or `neo4j`)
- **LLM**: `rag.llm.{ollama|openai|gemini|huggingface}` (default `ollama`)
- **API**: `api.host`, `api.port`

Paths (defaults):
- Processed chunks: `data/processed/chunked/<lang>`
- Mail threads: `data/processed/message_by_thread/<lang>`
- Persisted indexes: `data/processed/{faiss_index|chroma_db|bm25_index}`

## Run the API server
```bash
# From project root
python src/api/run.py
# or
(cd src && python api/run.py)

# Open docs:
# http://localhost:8000/docs
```

Root `/` serves `templates/index.html` (path controlled by `start_page` in config).

## Query the system
```bash
curl -sS -X POST 'http://localhost:8000/query' \
  -H 'Content-Type: application/json' \
  -d '{
        "question": "How do I use Boost.Asio for asynchronous networking?",
        "max_results": 5,
        "use_chat_history": true,
        "client_id": "demo"
      }'
```

Search-only (retrieval without generation):
```bash
curl -sS -X POST 'http://localhost:8000/search' \
  -H 'Content-Type: application/json' \
  -d '{"query": "Boost.Asio async_read", "max_results": 10}'
```

Get stats and configuration:
- `GET /rag/stats`
- `GET /config/rag`
- `GET /models/llm`, `GET /models/embedding`

## Ingest data
Trigger scraping + processing via API:
```bash
curl -sS -X POST 'http://localhost:8000/scrape' \
  -H 'Content-Type: application/json' \
  -d '{
        "source_url": "https://www.boost.org/doc/libs/latest/doc/html/boost_asio.html",
        "max_depth": 2,
        "delay": 1.0,
        "max_files": 50
      }'
```

Mailing list graph endpoints:
- `POST /maillist/thread/new`
- `POST /maillist/thread/email`
- `POST /maillist/messages/new`
- `PUT  /maillist/message/update`
- `DELETE /maillist/message/delete`

See request body schemas in the OpenAPI docs.

## CLI notes
`src/main_pipeline.py` exposes the orchestrator and an evaluator. Current default execution runs the simple evaluator:
```bash
python src/main_pipeline.py
```

## Tips
- Default embedding: `sentence-transformers/all-MiniLM-L6-v2` (offline friendly)
- FAISS is default; switch to Chroma via `rag.database.default_db_type`
- To change LLMs, update `rag.llm.default_llm_type` and model name

## Contributing
1) Fork 2) Branch 3) Implement 4) Test 5) PR

## License
MIT — see `src/LICENSE`.

## Support
- API usage: `src/api/POST_API_Guide.md`
- Configuration: `config/config.yaml`
- Orchestrator: `src/main_pipeline.py`
