# RAG to Django Migration - Master Plan

## Overview

This is the master plan for migrating the RAG system from a separate FastAPI service into the Django website-v2 backend.

### Priority Order

1. Task 1: Community Page Weekly Summary(START HERE)

   - RAG-by-topic approach to reduce hallucination
   - Weekly summary on Community page
   - Runs Sunday via Celery, reviewed before publishing

2. Task 2: Button-Triggered Library Summaries(To be created)

   - Users click button to get AI summaries
   - Discussion summaries, Historical context, FAQ generation

3. Task 3: AI-Generated FAQs (To be created)

   - Lowest-hanging fruit after community summary
   - FAQ generation for libraries

4. Task 4: Historical Section (To be created)

   - Library acceptance process, major decisions, concerns
   - Separate section on library homepage

5. Task 5: Semantic Search (To be created)

   - Third search option or replace all search

6. Task 6: Wagtail Review Interface (To be created)
   - PR-like approval workflow
   - FSC committee and general management review

---

## Architecture Overview

Django App: Create dedicated `rag/` app
Important: RAG will extend to documentation, Slack data and so on.

- `rag/` app: All RAG functionality (mailing list, documentation, Slack, extensible)
  - Core RAG components (pipeline, classifier, LLM helper, retriever, etc.)
  - Data source processors (mailing_list, documentation, slack)
  - Service wrapper (`RAGService`) that other apps can import

Directory Structure:

```
website-v2/
├── rag/  # Dedicated RAG app
│   ├── core/         # Core RAG components
│   ├── vector_data/  # vectorized data
│   ├── processors/   # Post-processors
│   ├── services.py   # RAGService (imported by other apps)
│   └── tasks.py      # Celery tasks
└── mailing_list/     # Mailing list app (uses RAG service)
```

---

## Migration Strategy

### Principles

1. MVP First: Quick MVP with mailing list data, iterate based on feedback
2. Direct Migration: No API service maintenance during migration
3. Incremental Migration: Migrate component by component, prioritize MVP features
4. Data Preservation: Ensure ChromaDB and email data remain intact
5. Human Review Required: All AI-generated content requires human review before publishing
6. S3 Backup: Vector data (ChromaDB) should be automatically saved to S3 whenever it is updated

---

## Key Decisions Needed

- Community summary timeframe: 7 days or 30 days?
- Semantic search: third option or replace all?
- Wagtail MVP: Consult Greg (estimated 1 week)

---

## Data Migration

### Step 1: Migrate ChromaDB

### Step 2: Configure Email Data Source

After migration, email data comes from PostgreSQL database (HyperKitty), not JSON files.

### Step 3: Update Configuration

Copy config files and update paths for Django project structure.

### Step 4: S3 Backup for Vector Data

Requirement: Vector data (ChromaDB) must be automatically saved to S3 whenever it is updated.

Implementation:

- Set up S3 bucket for vector data backups
- Implement automatic backup mechanism that triggers on ChromaDB updates
- Backup should include:
  - ChromaDB database files (chroma.sqlite3, collection data)
  - Embedding indices and metadata
  - Version/timestamp tracking for backups
- Consider incremental backups for efficiency
- Ensure backup happens asynchronously (Celery task) to avoid blocking updates

Backup Triggers:

- When new documents are added to vector store
- When documents are updated or deleted
- Scheduled periodic backups (e.g., daily) as safety net
- Manual backup trigger for major updates

Storage Strategy:

- Use versioned S3 bucket or timestamped backup files
- Keep multiple backup versions (e.g., last 7 days daily, then weekly)
- Compress backups to reduce storage costs

---
