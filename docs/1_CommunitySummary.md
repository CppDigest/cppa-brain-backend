# Task 1: Community Page Weekly Summary

## Overview

Implement weekly community page summary using RAG-by-topic approach. This is the **first task** to implement in the MVP.

**Status**: 🎯 **START HERE**

## Requirements (MVP)

- Weekly or monthly summary of mailing list discussions on Community page (not per-library)
- Runs Sunday via Celery, creates Wagtail draft for review
- Reviewed by FSC committee and general management before publishing
- Uses RAG to reduce hallucination by grounding summaries in retrieved discussions

## Implementation Strategy: RAG by Topic

### Core Approach

1. **Extract Topics**: Identify main topics from recent discussions (last 7 days)
2. **RAG Retrieval per Topic**: For each topic, retrieve relevant past discussions using RAG
3. **Chronological Summarization**: Summarize each topic's discussions chronologically to show evolution
4. **Grounded Summaries**: All summaries grounded in retrieved discussions (reduces hallucination)

### Why This Approach?

- ✅ **Reduces hallucination**: All summaries grounded in retrieved discussions (not just LLM knowledge)
- ✅ **Shows topic evolution**: Chronological summarization provides context and continuity
- ✅ **Relevant retrieval**: RAG finds past relevant discussions that might be missed with date-only filtering
- ✅ **True RAG**: Uses retrieval-augmented generation properly (retrieval + generation)

## Essential Components

- ✅ **HyperKitty database connection** (query recent emails for topic extraction)
- ✅ **RAG pipeline** (for retrieval - uses `RAGService.retrieve()` method)
  - RAG handles embeddings, similarity search, and retrieval automatically
  - Can use ChromaDB (recommended) or in-memory embeddings (for MVP)
- ✅ **LLM agent** (for topic extraction and chronological summarization)
- ✅ **Topic-based RAG retrieval** (RAG's core strength - retrieve relevant past discussions)
- ✅ **Chronological processing** (sort retrieved discussions by date for timeline)

## Implementation Options

### Option 1: Minimal MVP (Start Here)

- Use RAG pipeline with in-memory embeddings or simple ChromaDB setup
- RAG's `retrieve()` method handles all retrieval logic
- Fast to implement with existing RAG infrastructure

### Option 2: Full RAG (Recommended)

- ChromaDB vector store for faster similarity search
- Hybrid retriever (BM25 + embeddings) in RAG pipeline
- Better performance for large datasets

## Implementation Steps

### Step 1: Extract Topics from Recent Discussions

```python
# rag/tasks.py - Topic-based RAG implementation
from celery import shared_task
from django.utils import timezone
from datetime import timedelta, datetime
from django.db import connections
from .core.llm_agent import HuggingFaceAgent  # Or OpenAIAgent
from .services import RAGService

@shared_task
def generate_weekly_community_summary():
    """
    Generate weekly community summary using RAG by topic

    Strategy:
    1. Query recent discussions from HyperKitty database
    2. Extract main topics from recent discussions using LLM
    3. For each topic, use RAG to retrieve relevant past discussions
    4. Summarize each topic chronologically with retrieved context
    """
    # 1. Query recent emails (last 7 days) for topic identification
    date_end = timezone.now()
    date_start = date_end - timedelta(days=7)

    hyperkitty_db = connections['hyperkitty']

    recent_emails = []
    with hyperkitty_db.cursor() as cursor:
        cursor.execute("""
            SELECT
                e.subject,
                e.content,
                e.date,
                e.message_id,
                e.url
            FROM hyperkitty_email e
            WHERE e.date >= %s AND e.date <= %s
            ORDER BY e.date DESC
            LIMIT 200
        """, [date_start, date_end])

        for row in cursor.fetchall():
            recent_emails.append({
                'subject': row[0],
                'content': row[1],
                'date': row[2],
                'message_id': row[3],
                'url': row[4]
            })

    # 2. Extract main topics from recent discussions using LLM
    llm_agent = HuggingFaceAgent()

    # Combine recent discussions for topic extraction
    recent_text = "\n".join([
        f"[{e['date']}] {e['subject']}\n{e['content'][:300]}"
        for e in recent_emails[:50]  # Sample for topic extraction
    ])

    topics = llm_agent.extract_main_topics(recent_text)
    # Returns: ['Boost.Asio performance', 'Beast HTTP parser bug', 'Library proposal review', ...]
```

### Step 2: RAG Retrieval per Topic

```python
    # 3. For each topic, retrieve relevant past discussions using RAG
    from .services import RAGService

    # Initialize RAG service (uses full pipeline with ChromaDB or in-memory embeddings)
    rag_service = RAGService.get_pipeline()

    summaries_by_topic = {}

    for topic in topics[:10]:  # Top 10 topics
        # 3a. Use RAG to retrieve relevant past discussions for this topic
        # RAG will handle embeddings, similarity search, and retrieval
        # No date limit for retrieval - get all relevant past discussions

        retrieved_docs = rag_service.retrieve(
            question=topic,  # Use topic as query
            fetch_k=30,  # Get top 30 most relevant discussions
            filter_types=["mail"],  # Only search in mailing list
            str_results=False  # Return Document objects with metadata
        )

        # 3b. Convert retrieved documents to discussion format
        relevant_discussions = []
        for doc in retrieved_docs:
            # Extract metadata and content from Document
            doc_date = doc.metadata.get('date')
            if doc_date:
                # Convert string date to datetime if needed
                if isinstance(doc_date, str):
                    from dateutil import parser
                    try:
                        doc_date = parser.parse(doc_date)
                    except:
                        doc_date = None

            relevant_discussions.append({
                'subject': doc.metadata.get('subject', ''),
                'content': doc.page_content,  # Full document content
                'date': doc_date,
                'message_id': doc.metadata.get('message_id', ''),
                'url': doc.metadata.get('url', ''),
                'thread_id': doc.metadata.get('thread_id', ''),
                'relevance_score': doc.metadata.get('final_score', doc.metadata.get('score', 0.0)),
                'from_past': True  # Mark as retrieved from past (not in recent period)
            })

        # 3c. Add recent discussions for this topic (if any)
        # Find recent discussions that match this topic using simple keyword matching
        recent_for_topic = []
        topic_keywords = topic.lower().split()

        for email in recent_emails:
            email_text = f"{email['subject']} {email['content']}".lower()
            # Check if email is relevant to this topic
            if any(keyword in email_text for keyword in topic_keywords):
                recent_for_topic.append({
                    'subject': email['subject'],
                    'content': email['content'],
                    'date': email['date'],
                    'message_id': email['message_id'],
                    'url': email['url'],
                    'thread_id': None,
                    'relevance_score': 1.0,  # Recent discussions have high relevance
                    'from_past': False  # Mark as recent
                })

        # 3d. Combine recent + retrieved past discussions
        all_discussions = recent_for_topic + relevant_discussions

        # Remove duplicates by URL
        seen_urls = set()
        unique_discussions = []
        for disc in all_discussions:
            if disc['url'] and disc['url'] not in seen_urls:
                seen_urls.add(disc['url'])
                unique_discussions.append(disc)

        # 3e. Sort chronologically for timeline summarization
        unique_discussions.sort(key=lambda x: x['date'] if x['date'] else datetime.min, reverse=False)

        # 4. Summarize this topic chronologically using LLM
        topic_summary = llm_agent.summarize_topic_chronologically(
            topic=topic,
            discussions=unique_discussions,
            date_range=(date_start, date_end)
        )

        summaries_by_topic[topic] = {
            'summary': topic_summary,
            'discussion_count': len(unique_discussions),
            'recent_count': len(recent_for_topic),
            'retrieved_count': len(relevant_discussions),
            'date_range': {
                'first': unique_discussions[0]['date'].isoformat() if unique_discussions and unique_discussions[0]['date'] else None,
                'last': unique_discussions[-1]['date'].isoformat() if unique_discussions and unique_discussions[-1]['date'] else None
            },
            'sources': [d['url'] for d in unique_discussions[:5] if d['url']]  # Top 5 sources
        }

    # 5. Create final summary structure
    result = {
        'summary_by_topic': summaries_by_topic,
        'overall_stats': {
            'topics_count': len(summaries_by_topic),
            'recent_emails': len(recent_emails),
            'date_range': {
                'start': date_start.isoformat(),
                'end': date_end.isoformat()
            }
        },
        'ai_generated': True,
        'warning': "This summary is AI-generated from retrieved discussions. Please verify accuracy."
    }

    return result
```

### Step 3: Add LLM Methods for Topic Extraction and Summarization

```python
# rag/core/llm_agent.py - Add methods for topic-based RAG summarization
class HuggingFaceAgent(LLMAgent):
    # ... existing code ...

    def extract_main_topics(self, recent_discussions: str) -> List[str]:
        """
        Extract main topics from recent discussions
        Returns list of topic names
        """
        prompt = f"""Analyze these Boost C++ mailing list discussions and identify the main topics.
Return 5-10 distinct topics. Each topic should be specific (e.g., "Boost.Asio HTTP performance" not just "performance").

Discussions:
{recent_discussions[:3000]}

Return as JSON:
{{
  "topics": ["topic1", "topic2", "topic3", ...]
}}
"""

        try:
            # Use OpenAI or HuggingFace to extract topics
            if OPENAI_AVAILABLE and hasattr(self, 'client'):
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Extract main topics. Respond with JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                import json
                result = json.loads(response.choices[0].message.content)
                return result.get('topics', [])[:10]
            else:
                # Fallback: Use keyword extraction or simple clustering
                return self._extract_topics_keywords(recent_discussions)
        except Exception as e:
            self.logger.error(f"Topic extraction failed: {e}")
            return []

    def summarize_topic_chronologically(
        self,
        topic: str,
        discussions: List[dict],
        date_range: tuple
    ) -> str:
        """
        Summarize a topic's discussions chronologically
        Shows evolution over time (RAG-grounded summary)
        """
        # Format discussions chronologically
        discussions_text = ""
        for i, disc in enumerate(discussions):
            discussions_text += f"""
[{disc['date'].strftime('%Y-%m-%d')}] {disc['subject']}
{disc['content'][:400]}
URL: {disc['url']}
---
"""

        prompt = f"""Summarize these Boost C++ mailing list discussions about: {topic}

The discussions are ordered chronologically. Show how the topic evolved over time.
Focus on:
- Initial question or issue
- Responses and discussion points
- Resolution or current status
- Key decisions made

Discussions (in chronological order):
{discussions_text[:5000]}

Provide a chronological summary showing the evolution of this topic.
"""

        try:
            if OPENAI_AVAILABLE and hasattr(self, 'client'):
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Summarize chronologically. Be factual and reference specific discussions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            else:
                # Fallback: Simple chronological concatenation
                return self._simple_chronological_summary(discussions)
        except Exception as e:
            self.logger.error(f"Chronological summarization failed: {e}")
            return f"Topic: {topic} - {len(discussions)} relevant discussions found."
```

### Step 4: Celery Beat Schedule

```python
# website-v2/celery.py
app.conf.beat_schedule = {
    'generate-weekly-community-summary': {
        'task': 'rag.tasks.generate_weekly_community_summary',
        'schedule': crontab(hour=0, minute=0, day_of_week=6),  # Sunday 00:00
    },
}
```

### Step 5: Wagtail Integration

Create Wagtail page draft for review:

```python
# rag/tasks.py - Add Wagtail page creation
from .wagtail_hooks import AISummaryPage

# In generate_weekly_community_summary():
summary_page = AISummaryPage(
    title=f"Weekly Community Summary - {date_start.strftime('%B %d, %Y')}",
    summary_type='discussion',
    ai_content=result,
    status='pending_review',  # Requires review before publishing
    slug=f'community-summary-{date_start.strftime("%Y-%m-%d")}'
)
summary_page.save()

# Notify FSC committee and general management for review
# TODO: Send notification email
```

## Frontend Display

```django
<!-- templates/community/index.html -->
<div class="weekly-summary">
    {% if weekly_summary %}
        <div class="card">
            <div class="card-header">
                <h3>Weekly Community Summary</h3>
                <small class="text-muted">
                    <span class="ai-badge">🤖 AI-Generated</span>
                    Last updated: {{ weekly_summary.last_updated }}
                </small>
            </div>
            <div class="card-body">
                {% for topic, summary in weekly_summary.summary_by_topic.items %}
                    <div class="topic-summary">
                        <h4>{{ topic }}</h4>
                        <p>{{ summary.summary }}</p>
                        <small class="text-muted">
                            {{ summary.discussion_count }} discussions
                            ({{ summary.date_range.first }} to {{ summary.date_range.last }})
                        </small>
                    </div>
                {% endfor %}
            </div>
            <div class="card-footer">
                <small class="text-warning">
                    ⚠️ This summary is AI-generated using RAG retrieval. Please verify information.
                </small>
            </div>
        </div>
    {% endif %}
</div>
```

## Testing Checklist

- [ ] HyperKitty database connection works
- [ ] Topic extraction from recent discussions
- [ ] RAG retrieval finds relevant past discussions
- [ ] Chronological sorting works correctly
- [ ] LLM summarization produces coherent summaries
- [ ] Celery task runs on schedule (Sunday)
- [ ] Wagtail page creation for review
- [ ] Notification to FSC committee works
- [ ] Frontend displays summary correctly
- [ ] Review workflow functions properly

## Timeline

- **Week 1**: Set up database connection, topic extraction, basic retrieval
- **Week 2**: Implement chronological summarization, Celery task, Wagtail integration
- **Week 3**: Testing, refinement, review workflow

**Total**: 2-3 weeks

## Next Steps After This Task

Once community summary works, proceed to:

- Task 2: Button-Triggered Library Summaries
- Task 3: AI-Generated FAQs
- Task 4: Historical Section on Library Homepage
