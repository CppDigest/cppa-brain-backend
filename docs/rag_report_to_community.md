# RAG Service Integration Plan for Community Page

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Part 1: RAG System Design](#part-1-rag-system-design)
   - [1.1 Architecture Overview](#11-architecture-overview)
   - [1.2 Core Components](#12-core-components)
   - [1.3 Data Flow](#13-data-flow)
   - [1.4 Technology Stack](#14-technology-stack)
3. [Part 2: Django Project Integration](#part-2-django-project-integration)
   - [2.1 Integration Points](#21-integration-points)
   - [2.2 Database Schema](#22-database-schema)
   - [2.3 Celery Tasks](#23-celery-tasks)
   - [2.4 Views and Templates](#24-views-and-templates)
   - [2.5 Management Commands](#25-management-commands)
4. [Testing Strategy](#testing-strategy)
   - [4.1 Test Module Structure](#41-test-module-structure)
   - [4.2 Test Coverage Details](#42-test-coverage-details)
   - [4.3 Integration Testing](#43-integration-testing)

---

## Executive Summary

This document outlines the plan to integrate a Retrieval-Augmented Generation (RAG) system into the Django project to automatically generate and display AI-powered summaries of Boost mailing list discussions on the community page (`/community/`).

**Key Objectives:**
- Automatically generate weekly summaries of mailing list discussions
- Display summaries on the community page with proper formatting
- Maintain data synchronization between HyperKitty, ChromaDB, and PostgreSQL
- Ensure reliability through comprehensive testing

**Expected Benefits:**
- Improved user experience with summarized discussions
- Reduced manual curation effort
- Consistent, AI-powered content generation
- Scalable architecture for future enhancements

---

## Part 1: RAG System Design

### 1.1 Architecture Overview

The RAG system will be built as a modular service (`rag_service`) that operates independently but integrates seamlessly with the Django application.

```
┌─────────────────┐
│  HyperKitty DB  │ (Source of truth for emails)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │ (Vector embeddings for semantic search)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │ (Retrieval + Generation)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PostgreSQL     │ (Structured summary storage)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Django Views   │ (Display on /community/)
└─────────────────┘
```

**Key Design Principles:**
- **Modularity:** Each component has a single responsibility
- **Separation of Concerns:** RAG logic separate from Django views
- **Scalability:** Support for multiple data sources and retrieval methods
- **Testability:** Comprehensive test coverage for all components

### 1.2 Core Components

#### 1.2.1 RAG Pipeline (`rag_service/langchain_rag/rag_pipeline.py`)

Main orchestrator for RAG operations that initializes retrievers, processes queries, and manages document CRUD operations. Provides methods for retrieval (`retrieve`), full RAG queries (`query`), and document management (`add_mail_data`, `update_mail_data`, `delete_document`).

#### 1.2.2 Data Processors (`rag_service/langchain_rag/preprocessor/`)

**`mail_preprocessor.py`:** Processes email data into LangChain Documents by extracting metadata, converting dates to timestamps, chunking large emails, and generating document IDs. 
**`docu_preprocessor.py`:** Processes documentation files by parsing formats, extracting structured content, and generating embeddings-ready documents.

#### 1.2.3 Retrievers (`rag_service/langchain_rag/retrieve/`)

**`hybrid_retriever.py`:** Combines dense (semantic similarity) and sparse (BM25 keyword-based) retrieval with result fusion and ranking. 
**`multi_base_retriever.py`:** Manages multiple data source retrievers (one per type) with a unified query interface and filter-based result combination.

#### 1.2.4 LLM Integration (`rag_service/langchain_rag/llm/`)

**`llm_helper.py`:** Main LLM orchestration for prompt construction, response parsing, and metadata extraction (categories, sentiment, libraries). 
**`openai_agent.py`:** OpenAI API integration with GPT model support, streaming responses, and token management. 
**`huggingface_agent.py`:** HuggingFace model support for local inference using Transformers library with custom model configuration.

#### 1.2.5 Task Modules (`rag_service/langchain_rag/task/`)

**`community_task.py`:** Contains `WeeklyCommunitySummaryGenerator` for orchestrating topic extraction (thread-based, clustering, or LLM) and summary generation per topic. 
**`mail_data_retriever.py`:** Provides unified mail data access with HyperKitty PostgreSQL integration, ChromaDB query interface, and automatic fallback between sources. 
**`topic_extractor.py`:** Implements topic extraction algorithms including thread-based grouping, semantic clustering, and LLM-based extraction. 
**`metadata_validate_modify.py`:** Handles metadata validation, normalization, missing field completion, and data quality checks.

### 1.3 Data Flow

#### 1.3.1 Email Collection Flow

```
HyperKitty PostgreSQL
    │
    ├─→ MailDataRetriever.get_mails()
    │   └─→ SQL Query (date range, limit)
    │
    ├─→ MailPreprocessor.process_mail_list()
    │   ├─→ Extract metadata
    │   ├─→ Convert to Documents
    │   └─→ Chunk if needed
    │
    ├─→ RAGPipeline.add_mail_data()
    │   ├─→ Check for duplicates
    │   ├─→ Generate embeddings
    │   └─→ Store in ChromaDB
    │
    └─→ ChromaDB (Vector Store)
```

#### 1.3.2 Summary Generation Flow

```
WeeklyCommunitySummaryGenerator
    │
    ├─→ 1. Get Recent Emails
    │   └─→ MailDataRetriever.get_mails(start_date, end_date)
    │
    ├─→ 2. Extract Topics
    │   ├─→ TopicExtractor.extract_topics()
    │   └─→ Method: thread/clustering/llm
    │
    ├─→ 3. For Each Topic
    │   ├─→ RAGPipeline.retrieve(topic_query, fetch_k=30)
    │   │   └─→ HybridRetriever.retrieve()
    │   │       ├─→ Dense retrieval (embeddings)
    │   │       ├─→ Sparse retrieval (BM25)
    │   │       └─→ Result fusion
    │   │
    │   ├─→ LLMHelper.process_pipeline(retrieved_docs, topic)
    │   │   ├─→ Construct prompt
    │   │   ├─→ Call LLM API
    │   │   └─→ Parse response
    │   │
    │   └─→ Generate assertions + chronological summary
    │
    └─→ 4. Aggregate Results
        └─→ Save to CommunitySummary model
```

#### 1.3.3 Display Flow

```
User visits /community/
    │
    ├─→ CommunitySummaryView.get_context_data()
    │   ├─→ Query: CommunitySummary.objects.filter(need_review=False)
    │   ├─→ Normalize URLs
    │   ├─→ Assign reference numbers
    │   └─→ Prepare template context
    │
    └─→ community.html template
        ├─→ Render topics
        ├─→ Display assertions
        ├─→ Show chronological summaries
        └─→ Display reference URLs
```

### 1.4 Technology Stack

**Core Technologies:**
- **LangChain:** Framework for RAG pipeline orchestration
- **ChromaDB:** Vector database for embeddings storage
- **OpenAI API / HuggingFace:** LLM providers
- **sentence-transformers:** Embedding models
- **BM25:** Sparse retrieval algorithm

**Python Libraries:**
- `langchain-core`: Core LangChain abstractions
- `chromadb`: Vector database client
- `openai`: OpenAI API client
- `transformers`: HuggingFace model support
- `rank-bm25`: BM25 implementation
- `numpy`, `pandas`: Data processing

**Django Integration:**
- Django ORM for database operations
- Celery for asynchronous tasks
- Django management commands for manual operations

---

## Part 2: Django Project Integration

### 2.1 Integration Points

#### 2.1.1 Django App Structure

The RAG service will be integrated as a Django app:

```
rag_service/
├── __init__.py
├── apps.py                    # Django app configuration
├── models.py                  # CommunitySummary model
├── views.py                   # CommunitySummaryView
├── tasks.py                   # Celery tasks
├── services.py                # RAGService singleton
├── admin.py                   # Django admin interface
├── s3_utils.py                # S3 backup utilities
├── management/commands/
│   └── generate_community_summary.py
├── langchain_rag/            # RAG implementation
└── tests/                     # Test suite
```

#### 2.1.2 Settings Configuration

**Required Settings:**
```python
# config/settings.py

# RAG Configuration
RAG_CONFIG = {
    'chroma_persist_dir': 'chroma_db',
    'enable_cache': True,
    'enable_telemetry': False,
    'base_retriever_types': ['mail'],
}

# LLM Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# S3 Configuration (for backups)
RAG_S3_BUCKET_NAME = os.getenv('RAG_S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
```

#### 2.1.3 URL Configuration

```python
# config/urls.py
urlpatterns = [
    # ... existing patterns
    path('community/', CommunitySummaryView.as_view(), name='community'),
]
```

### 2.2 Database Schema

The `CommunitySummary` model stores AI-generated weekly summaries with the following key concepts:

- **Review Workflow:** `need_review` flag controls display (Wagtail sets to `False` after review)
- **Data Storage:** `summary_data` contains publishable content, `original_summary_data` preserves the raw AI-generated version (read-only)
- **Metadata:** `model_info` stores AI model details, `topics_count` and `recent_emails_count` provide quick statistics
- **Review Tracking:** `main_reviewer` and `last_modified_at` track review history

#### 2.2.1 Summary Data Structure

The `summary_data` field stores only `summary_by_topic`:

```json
{
  "summary_by_topic": [
    {
      "subject": "Topic Name",
      "assertions": [
        {
          "content": "Key point extracted from discussions",
          "reference_url": ["url1", "url2"]
        }
      ],
      "chronological_summary": [
        {
          "date": "2025-11-06",
          "summary": "What happened on this date",
          "reference_url": ["url1"]
        }
      ]
    }
  ]
}
```

### 2.3 Celery Tasks

#### 2.3.1 Email Synchronization Task

**Task:** `sync_new_mails_to_vector_db()`

**Schedule:** Every 30 minutes

**Purpose:** Keep ChromaDB synchronized with HyperKitty

**Error Handling:**
- Retry on connection failures
- Log failed emails for manual review
- Continue processing on individual failures

#### 2.3.2 Summary Generation Task

**Task:** `update_summary_data()`

**Schedule:** Every Sunday at 1:00 AM

**Purpose:** Generate weekly community summaries

#### 2.3.3 S3 Backup Task

**Task:** `upload_vector_data_to_s3()`

**Schedule:** Once per day

**Purpose:** Backup ChromaDB vector data to S3 for disaster recovery

**Error Handling:**
- Log upload failures for manual intervention
- Continue operation even if backup fails (non-critical)

### 2.4 Views and Templates

#### 2.4.1 CommunitySummaryView

**Location:** `rag_service/views.py`

**Purpose:** Display community summaries on `/community/`


#### 2.4.2 Template Structure

**Location:** `templates/community.html`

**Sections:**
1. **Header:** Summary period, statistics
2. **Topics:** List of discussion topics
3. **Assertions:** Key points per topic
4. **Chronological Summary:** Timeline of discussions
5. **References:** Numbered URL references

**Features:**
- Responsive design
- Collapsible sections
- Reference number links
- Empty state handling

### 2.5 Management Commands

#### 2.5.1 generate_community_summary

**Purpose:** Manual summary generation

**Options:**
- `--test`: Generate test data (no LLM calls). When used, sets `need_review=False` so the summary displays immediately.

**Usage:**
```bash
# Generate test summary (displays immediately)
python manage.py generate_community_summary --test

# Generate real summary (requires Wagtail review)
python manage.py generate_community_summary
```

#### 2.5.2 Additional Commands (Future)

- `sync_mails`: Manual email synchronization
- `reindex_chromadb`: Rebuild vector index
- `validate_summaries`: Check summary data quality

---

## Testing Strategy

### 4.1 Test Module Structure

The test suite is organized into logical modules for maintainability and clarity:

```
rag_service/tests/
├── conftest.py                 # Shared fixtures
├── fixtures.py                 # Test data fixtures
├── test_models.py              # Model tests
├── test_views.py               # View tests
├── test_tasks.py               # Celery task tests
├── test_commands.py            # Management command tests
└── tests_rag_pipeline/         # RAG pipeline integration tests
    ├── conftest.py             # Pipeline-specific fixtures
    ├── test_pipeline_basic.py      # Initialization & basic operations
    ├── test_pipeline_operations.py # CRUD operations
    ├── test_pipeline_errors.py     # Error handling scenarios
    └── test_pipeline_validation.py # Data validation & edge cases
```

### 4.2 Test Coverage Details

#### 4.2.1 Model Tests (`test_models.py`)

Tests `CommunitySummary` model creation, field validation, database constraints, and query methods. Verifies JSON field structure, timestamp handling, and ordering by `generated_at`.

#### 4.2.2 View Tests (`test_views.py`)

Tests `CommunitySummaryView` context data preparation, URL normalization, and reference number assignment. Verifies active summary retrieval, empty state handling, and data transformation for templates.

#### 4.2.3 Task Tests (`test_tasks.py`)

Tests Celery tasks including email synchronization (`sync_new_mails_to_vector_db`) and summary generation (`generate_weekly_community_summary`, `update_summary_data`). Covers success scenarios, error handling, network failures, and invalid data.

#### 4.2.4 Command Tests (`test_commands.py`)

Tests management command execution, option parsing (`--test`), and command output validation. Verifies test data generation and error message handling.

#### 4.2.5 RAG Pipeline Tests (`tests_rag_pipeline/`)

**Basic Tests (`test_pipeline_basic.py`):** Tests pipeline initialization, component setup, RAGService singleton pattern, and basic retrieve/query operations. Verifies end-to-end query pipeline flow.

**Operations Tests (`test_pipeline_operations.py`):** Tests document CRUD operations (`add_mail_data`, `update_mail_data`, `delete_document`), batch query processing, cache management, and statistics retrieval.

**Error Tests (`test_pipeline_errors.py`):** Tests network failure scenarios including LLM timeout/rate limit errors, ChromaDB connection failures, S3 upload failures, and retriever errors. Verifies graceful error handling and degradation.

**Validation Tests (`test_pipeline_validation.py`):** Tests invalid data handling, edge cases (empty inputs, missing fields, malformed data), and public method coverage. Verifies None/null handling, duplicate documents, and partial batch failures.

### 4.3 Integration Testing

#### 4.3.1 End-to-End Tests

**Full Pipeline Test:**
1. Sync emails from HyperKitty
2. Generate summary
3. Save to database
4. Display on community page

**Components Tested:**
- MailDataRetriever → RAGPipeline → LLMHelper
- WeeklyCommunitySummaryGenerator → CommunitySummary model
- CommunitySummaryView → Template rendering

#### 4.3.2 Mock Strategy

**External Dependencies Mocked:**
- HyperKitty database (SQL queries)
- ChromaDB (vector operations)
- LLM APIs (OpenAI, HuggingFace)
- S3 operations

**Benefits:**
- Fast test execution
- No external dependencies
- Predictable test results
- Isolated component testing

#### 4.3.3 Fixture Strategy

**Shared Fixtures (`conftest.py`):**
- `mock_pipeline_config`: Test configuration
- `sample_documents`: Sample LangChain Documents
- Database fixtures for models

**Test-Specific Fixtures:**
- Mock retriever instances
- Mock LLM responses
- Sample email data
- Summary JSON structures

---

## Conclusion

This plan provides a comprehensive roadmap for integrating RAG capabilities into the Django project's community page. The modular design ensures maintainability, while comprehensive testing ensures reliability.

**Success Criteria:**
- Automated weekly summary generation
- Reliable display on community page
- Comprehensive test coverage
- Performance within acceptable limits
- Positive user feedback

---

## To Be Discussed

### Wagtail Integration

**Handled separately by the Wagtail team**:

Review interface in Wagtail controls the `need_review` flag for each summary.
Any reviewer edits must be persisted back to the `CommunitySummary` database record before activation.



