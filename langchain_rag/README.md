# LangChain RAG Pipeline

High-performance hybrid retrieval system for the C++ Boost Assistant using LangChain and ChromaDB.

## 🚀 Features

- ✅ **Hybrid Retrieval**: Combines dense (ChromaDB vector) and sparse (BM25) retrieval
- ✅ **Query Caching**: Fast responses with TTL-based caching and LRU eviction
- ✅ **Performance Telemetry**: Built-in monitoring and metrics
- ✅ **Document Type Filtering**: Filter by type (documentation, mail, etc.)
- ✅ **Incremental Updates**: Add, update, delete documents without full reindex
- ✅ **Memory Optimized**: ~1.5-2GB RAM usage

---

## 📦 Installation

```bash
cd langchain_rag
pip install -r requirements.txt
```

---

## 🎯 Quick Start

```python
from langchain_rag import create_langchain_rag_pipeline

# Create pipeline
pipeline = create_langchain_rag_pipeline(
    mail_data_dir="data/processed/message_by_thread",
    doc_data_dir="data/source_data/processed/en",
    force_reindex=False,
)

# Retrieve relevant documents
docs = pipeline.retrieve(
    question="How does Boost.Asio handle asynchronous operations?",
    fetch_k=10
)

# With type filtering
docs = pipeline.retrieve(
    question="Boost smart pointers",
    fetch_k=10,
    filter_types=["documentation"]  # Only documentation
)

# Display results
for doc in docs[:3]:
    print(f"Score: {doc.metadata['final_score']:.3f}")
    print(f"Type: {doc.metadata['type']}")
    print(f"Content: {doc.page_content[:200]}...\n")
```

---

## ⚙️ Configuration

```python
from langchain_rag import LangChainConfig, LangChainRAGPipeline

config = LangChainConfig(
    # Data paths
    mail_data_dir="data/processed/message_by_thread",
    doc_data_dir="data/source_data/processed/en",
    
    # Embedding model
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    
    # Retrieval settings
    dense_top_k=100,        # ChromaDB results
    sparse_top_k=100,       # BM25 results
    final_top_k=10,         # Final returned
    
    # Chunking
    chunk_size=1024,
    chunk_overlap=100,
    
    # Performance
    enable_cache=True,
    enable_telemetry=True,
    force_reindex=False,
)

pipeline = LangChainRAGPipeline(config=config)
```

---

## 📊 Performance & Monitoring

```python
# Cache statistics
stats = pipeline.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")

# Telemetry report
pipeline.print_performance_report()

# Clear cache
pipeline.clear_cache()
```

## 🔄 Incremental Updates

```python
# Add new mail messages
new_messages = [{
    "message_id": "msg_001",
    "subject": "Question about Boost.Asio",
    "content": "How do I use async_read?",
    "sender_address": "user@example.com",
    "date": "2025-10-27",
    "url": "https://lists.boost.org/...",
}]

result = pipeline.add_mail_data(new_messages)
print(f"Added: {result['added_count']}")

# Update document
pipeline.update_mail_data(updated_message)

# Delete document
pipeline.delete_document("doc_id")
```

---
## 📁 Components

| File | Purpose |
|------|---------|
| `rag_pipeline.py` | Main orchestrator |
| `hybrid_retriever.py` | ChromaDB + BM25 hybrid search |
| `data_processor.py` | Document loading & chunking |
| `config.py` | Configuration management |
| `caching_telemetry.py` | Caching & performance monitoring |

**How It Works:**
1. **Dense retrieval**: ChromaDB vector similarity search
2. **Sparse retrieval**: BM25 keyword search (dynamic)
3. **Reranking**: TF-IDF + time bonus for recent emails
4. **Caching**: Query results cached with TTL

---

## 🤝 API Integration

```python
# api/__init__.py
from langchain_rag import LangChainRAGPipeline, LangChainConfig

rag_pipeline = LangChainRAGPipeline(config=LangChainConfig.from_env())

@app.post("/query")
def query(request: QueryRequest):
    docs = rag_pipeline.retrieve(
        question=request.question,
        fetch_k=request.search_limit,
        filter_types=request.search_scopes,
    )
    return {"results": docs, "count": len(docs)}
```

---

## 📚 Documentation

See also:
- `api/API_Report.md` - API documentation

---

## 🙏 Built With

- [LangChain](https://python.langchain.com/) - Framework
- [ChromaDB](https://www.trychroma.com/) - Vector store
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Loguru](https://github.com/Delgan/loguru) - Logging

---

Part of the C++ Boost Assistant project.
