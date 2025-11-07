# Task 1: Community Page Weekly Summary

## Overview

Implement weekly community page summary using RAG-by-topic approach. This is the first task to implement in the MVP.

Status: START HERE

## Requirements

- Weekly or monthly summary of mailing list discussions on Community page
- Runs Sunday via Celery, creates Wagtail draft for review
- Reviewed by FSC committee and general management before publishing
- Uses RAG to reduce hallucination by grounding summaries in retrieved discussions

## Migration Steps

### Step 1: Extract Topics from Recent Discussions

- Query recent emails from HyperKitty database (last 7 days)
- Extract main topics from recent discussions using LLM agent
- Returns list of topic names

### Step 2: RAG Retrieval per Topic

- For each topic, use RAG service to retrieve relevant past discussions
- Use `RAGService.retrieve()` method with topic as query
- Filter by mail type, fetch top 30 relevant discussions
- Combine recent discussions with retrieved past discussions
- Sort chronologically for timeline summarization

### Step 3: Celery Task Setup

- Create `generate_weekly_community_summary()` Celery task
- Schedule task to run Sunday 00:00 via Celery Beat
- Task generates summary and creates Wagtail page draft

### Step 4: Wagtail Integration

- Create AISummaryPage Wagtail model
- Task creates Wagtail page draft with status 'pending_review'
- Notify FSC committee and general management for review

## Essential Components

- HyperKitty database connection
- RAG pipeline (ChromaDB or in-memory embeddings)
- LLM agent (topic extraction and summarization)
- Celery task scheduler
- Wagtail CMS integration

## Timeline

- Week 1: Set up database connection, topic extraction, basic retrieval
- Week 2: Implement chronological summarization, Celery task, Wagtail integration
- Week 3: Testing, refinement, review workflow

Total: 2-3 weeks
