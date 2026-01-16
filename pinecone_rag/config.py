"""
Configuration file for LangChain + Pinecone + Neo4j RAG pipeline.

Replace placeholders with actual API keys and endpoints.
"""

from typing import Optional
from dataclasses import dataclass, field
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load environment variables from a local .env file if python-dotenv is available.
if load_dotenv is not None:
    load_dotenv()


@dataclass
class PineconeConfig:
    """Pinecone configuration"""

    IS_TEST = False

    api_key: str = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key-here")
    environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-central1")
    cloud: str = os.getenv("PINECONE_CLOUD", "gcp")

    if IS_TEST:
        api_key: str = os.getenv("TEST_PINECONE_API_KEY", "your-pinecone-api-key-here")
        environment: str = os.getenv("TEST_PINECONE_ENVIRONMENT", "us-east-1")
        cloud: str = os.getenv("TEST_PINECONE_CLOUD", "aws")

    index_name: str = os.getenv("PINECONE_INDEX_NAME", "rag-hybrid")
    # Note: dimension is automatically set by Pinecone when using integrated embeddings
    top_k: int = 5  # Default number of documents to retrieve
    chunk_size: int = int(
        os.getenv("PINECONE_CHUNK_SIZE", "1000")
    )  # Document chunk size
    chunk_overlap: int = int(
        os.getenv("PINECONE_CHUNK_OVERLAP", "200")
    )  # Chunk overlap
    batch_size: int = int(
        os.getenv("PINECONE_BATCH_SIZE", "96")
    )  # Batch size for upserting documents
    rerank_model = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")


@dataclass
class LLMConfig:
    """LLM configuration"""

    model_name: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    temperature: float = 0.0
    max_tokens: int = 1000
    api_key: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


@dataclass
class EmbeddingConfig:
    """Pinecone cloud embedding model configuration for hybrid search"""

    # Dense embedding model for semantic search
    pinecone_model: str = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
    # Sparse embedding model for keyword/lexical search
    pinecone_sparse_model: str = os.getenv(
        "PINECONE_SPARSE_EMBEDDING_MODEL", "pinecone-sparse-english-v0"
    )


@dataclass
class CacheConfig:
    """Cache configuration"""

    enabled: bool = True
    ttl_seconds: int = 86400  # 1 day in seconds
    max_size: int = 1000  # Maximum number of cached items (for LRU)
    use_disk_cache: bool = False  # Set to True for persistent cache
    cache_dir: str = "./cache"  # Directory for disk-based cache


@dataclass
class LoggingConfig:
    """Logging configuration"""

    log_file: str = "logs/retrieval.log"
    log_level: str = "INFO"
    include_timestamps: bool = True


@dataclass
class MailConfig:
    """Mail preprocessor configuration"""

    mail_data_dir: str = os.getenv("MAIL_DATA_DIR", "data/message_by_thread")
    namespace: str = os.getenv("MAIL_NAMESPACE", "mailing")


@dataclass
class DocuConfig:
    """Documentation preprocessor configuration"""

    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "data/cpp_documentation")
    md_data_dir: str = os.getenv("MD_DATA_DIR", "data/cpp_markdown_documentation")
    namespace: str = os.getenv("DOC_NAMESPACE", "cpp-documentation")


@dataclass
class SlackConfig:
    """Slack database configuration"""

    db_host: str = os.getenv("SLACK_DB_HOST", "localhost")
    db_port: int = int(os.getenv("SLACK_DB_PORT", "5432"))
    db_name: str = os.getenv("SLACK_DB_NAME", "CPPLang")
    db_user: str = os.getenv("SLACK_DB_USER", "postgres")
    db_password: str = os.getenv("SLACK_DB_PASSWORD", "")
    message_table: str = os.getenv("SLACK_TABLE_NAME", "slack_message")
    channel_table: str = os.getenv("SLACK_CHANNEL_TABLE", "slack_channel")
    team_table: str = os.getenv("SLACK_TEAM_TABLE", "slack_team")
    email_table: str = os.getenv("SLACK_EMAIL_TABLE", "email")
    user_table: str = os.getenv("SLACK_USER_TABLE", "profile")
    namespace: str = os.getenv("SLACK_NAMESPACE", "slack-Cpplang")


@dataclass
class RAGConfig:
    """Main RAG pipeline configuration"""

    # Use default_factory to avoid sharing the same instances across RAGConfig objects
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mail: MailConfig = field(default_factory=MailConfig)
    docu: DocuConfig = field(default_factory=DocuConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)

    # Metadata filtering example
    default_metadata_filter: Optional[dict] = None  # e.g., {"project": "RAG"}


# Global configuration instance
config = RAGConfig()
