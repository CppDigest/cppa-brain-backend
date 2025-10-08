# C++ Boost RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system designed specifically for C++ Boost library documentation and mailing list archives. This system provides intelligent question-answering capabilities by combining advanced document processing, semantic search, and large language models.

## 🚀 Features

### Core Capabilities
- **Multi-format Document Processing**: Handles HTML, JSON, text, and email archives
- **Semantic Chunking**: Intelligent document segmentation preserving context
- **Hybrid Retrieval**: Combines vector search, BM25, graph search, and hierarchical search
- **Multiple LLM Support**: Integration with OpenAI, Gemini, Ollama, and HuggingFace models
- **Real-time Processing**: Background data updates and incremental indexing
- **RESTful API**: Complete API for data ingestion and querying

### Advanced RAG Features
- **Cross-Encoder Reranking**: Improves retrieval quality with neural reranking
- **Multi-step Reasoning**: Complex query decomposition and iterative refinement
- **Self-reflection**: Built-in answer quality assessment
- **Hierarchical Search**: Email thread and document structure awareness
- **Graph-based Retrieval**: Knowledge graph construction and traversal
- **Context Filtering**: Intelligent relevance and redundancy filtering

## 📁 Project Structure

```
cppa-brain-backend/
├── api/                          # REST API endpoints
│   ├── vector_data_api.py       # Main API server
│   ├── chat_history_manager.py  # Conversation management
│   └── POST_API_Guide.md        # API documentation
├── config/                       # Configuration files
│   └── config.yaml              # System configuration
├── data_processor/               # Document processing modules
│   ├── multiformat_processor.py # Multi-format document handler
│   ├── semantic_chunker.py     # Semantic text chunking
│   ├── summarize_processor.py   # Document summarization
│   └── mail_json_processor.py   # Email archive processing
├── rag/                         # RAG system components
│   ├── improved_rag_system.py  # Main RAG orchestrator
│   ├── document_graph.py        # Knowledge graph management
│   ├── mail_hierarchical_rag.py # Email thread processing
│   ├── reranker.py             # Cross-encoder reranking
│   ├── evaluation_system.py    # RAG evaluation metrics
│   └── langchain/              # LangChain integration
├── text_generation/             # LLM integration modules
│   ├── llm_manager.py          # LLM orchestration
│   ├── openai_chatbot.py       # OpenAI integration
│   ├── gemini_chatbot.py       # Google Gemini integration
│   ├── ollama_chatbot.py       # Ollama local models
│   └── huggingface_chatbot.py # HuggingFace models
├── utils/                       # Utility modules
│   └── config.py               # Configuration management
├── templates/                   # Web interface templates
└── main_pipeline.py            # Main pipeline orchestrator
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd cppa-brain-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure the system
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your settings
```

### Dependencies
Key dependencies include:
- `fastapi` - Web framework
- `sentence-transformers` - Embedding models
- `faiss` / `chromadb` - Vector databases
- `langchain` - RAG framework
- `transformers` - HuggingFace models
- `loguru` - Logging
- `pydantic` - Data validation

## ⚙️ Configuration

The system is configured through `config/config.yaml`. Key configuration sections:

### Embedding Models
```yaml
rag:
  embedding:
    embedding_types_list: ["gemma", "minilm", "nomic", "jina", "baai"]
    default_embedding_type: "minilm"
```

### Vector Databases
```yaml
rag:
  database:
    db_types_list: ["faiss", "chroma"]
    default_db_type: "faiss"
```

### LLM Integration
```yaml
rag:
  llm:
    llm_types_list: ["gemini", "ollama", "openai", "huggingface"]
    default_llm_type: "ollama"
```

## 🚀 Usage

### Starting the Pipeline
```bash
# Run the main pipeline
python main_pipeline.py

# Run with specific configuration
python main_pipeline.py --config config/config.yaml --language en
```

### API Server
```bash
# Start the API server
python -m api.vector_data_api

# The API will be available at http://localhost:8000
```

### Web Interface
```bash
# Access the web interface
# Open templates/index.html in your browser
```

## 📡 API Endpoints

### Data Ingestion
- `POST /api/scrape` - Scrape and process new documents
- `POST /api/maillist/thread/new` - Add email thread data
- `POST /api/messages/thread/new` - Add email messages

### Query Interface
- `POST /api/query` - Query the RAG system
- `GET /api/status` - System status
- `GET /api/stats` - System statistics

### Example Query
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I use Boost.Asio for asynchronous networking?",
    "max_results": 5,
    "use_reranker": true
  }'
```

## 🔧 Advanced Features

### Multi-step Reasoning
The system supports complex query decomposition:
```python
# Enable multi-step reasoning
query_config = {
    "use_multi_step": True,
    "max_steps": 5,
    "confidence_threshold": 0.8
}
```

### Hierarchical Search
Email threads are processed with hierarchical structure:
- Thread-level context
- Message-level details
- Sender and relationship tracking

### Graph-based Retrieval
Knowledge graphs capture document relationships:
- Document similarity
- Concept relationships
- Cross-reference links

## 📊 Evaluation

The system includes comprehensive evaluation metrics:
- **Groundedness**: Answer accuracy to source documents
- **Faithfulness**: Consistency with retrieved context
- **Relevance**: Query-answer alignment
- **Completeness**: Answer thoroughness

## 🔄 Data Processing Pipeline

1. **Scraping**: Extract documents from Boost.org
2. **Processing**: Multi-format document parsing
3. **Chunking**: Semantic text segmentation
4. **Embedding**: Vector representation generation
5. **Indexing**: Vector and search index creation
6. **Graph Construction**: Knowledge graph building

## 🎯 Use Cases

- **Documentation Q&A**: Answer questions about Boost libraries
- **Code Examples**: Provide compilable code snippets
- **Mailing List Search**: Find relevant discussions and solutions
- **Learning Assistant**: Guide users through Boost concepts
- **Research Support**: Academic and professional research

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Check the API documentation in `api/POST_API_Guide.md`
- Review configuration options in `config/config.yaml`
- Examine the main pipeline in `main_pipeline.py`

## 🔮 Future Enhancements

- Real-time collaboration features
- Advanced graph analytics
- Multi-language support expansion
- Performance optimization
- Enhanced evaluation metrics
