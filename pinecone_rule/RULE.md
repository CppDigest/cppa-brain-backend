---
description: "Rules for querying Pinecone using hybrid search and generating systematic, synthesized reports with logical organization and proper citations"
alwaysApply: false
---

# Pinecone Query and Answer Generation Rule

When asked to retrieve information or answer questions using Pinecone vector database:

---

## Processing Procedures

### 1. Environment Setup

**Check and install required dependencies:**

- Check if `python-dotenv>=1.0.0` is installed
- Check if `pinecone>=3.0.0` is installed
- Check if `langchain-core>=0.1.0` is installed

- If not installed, run: `pip install python-dotenv>=1.0.0 pinecone>=3.0.0 langchain-core>=0.1.0`
- Ensure `.env` file exists with required Pinecone configuration:
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME` (default: "rag-hybrid")
  - `PINECONE_RERANK_MODEL` (default: "bge-reranker-v2-m3")

### 2. Extract Parameters for query.py

### 2.1 Query Text

- Extract and clean the core query text from the user's prompt
- Remove unnecessary words but preserve intent
- Pass as the `query` parameter to `PineconeQuery.query()`

### 2.2 Determine top_k

- Default: `10` documents
- Adjust based on user's request:
  - If user asks for "a few" or "several": use `5`
  - If user asks for "many" or "comprehensive": use `20`
  - If user specifies a number: use that number (within reasonable limits: 1-10000)
- Pass as `top_k` parameter

### 2.3 Determine Namespace

- Analyze user's question to determine the appropriate namespace:
  - **"mailing"**: For questions about Boost mailing list discussions, email threads, community discussions
  - **"slack-Cpplang"**: For questions about Slack conversations, Cpplang team discussions, chat history
  - **"wg21-papers"**: For questions about C++ standard proposals, WG21 papers
  - **"cpp-documentation"**: For questions about C++ documentation, Boost library documentation
- If namespace cannot be determined from context, default to **"mailing"**
- Pass as `namespace` parameter

### 2.4 Extract Metadata Filter

Extract metadata filters based on user's prompt and available metadata fields for each namespace.

#### Available Metadata Fields by Namespace:

**Namespace: "mailing"**
- `timestamp`: Unix timestamp (float) - email sent date
- `thread_id`: String - thread identifier
- `subject`: String - email subject line
- `author`: String - sender email address
- `parent_id`: String - parent message ID
- `doc_id`: String - message ID
- `type`: String - always "mailing"

**Namespace: "slack-Cpplang"**
- `timestamp`: Unix timestamp (float) - message timestamp
- `thread_ts`: String - thread timestamp (empty string if not in thread)
- `channel_id`: String - Slack channel ID
- `team_id`: String - Slack team ID
- `user_name`: String - user display name
- `is_grouped`: Boolean - whether message is grouped
- `group_size`: Integer - number of messages in group
- `doc_id`: String - message timestamp
- `type`: String - always "slack"

**Namespace: "wg21-papers"**
- `timestamp`: Unix timestamp (float) - paper date
- `document_number`: String - paper number (e.g., "P0843R10", "N1234")
- `title`: String - paper title
- `author`: String - author name
- `url`: String - paper URL
- `filename`: String - local filename
- `doc_id`: String - document identifier
- `type`: String - always "wg21-papers"

**Namespace: "cpp-documentation"**
- `build_time`: Unix timestamp (float) - documentation build time
- `library`: String - Source name ("cppreference.com", "isocpp.github.io", "gcc.gnu.org", "cplusplus.com", "git_MicrosoftDocs", "git_cplusplus")
- `lang`: String - language code (typically "en")
- `doc_id`: String - documentation URL
- `type`: String - always "documentation"

#### Filter Construction:

**1. Timestamp Filtering (Most Common)**

Look for time-related keywords in user's prompt and calculate appropriate date ranges:

- **"recent", "latest", "new", "current"** → filter for recent documents (last 30 days from today)
- **"last week"** → calculate date range for the previous calendar week (Monday to Sunday)
- **"last month"** → calculate date range for the **previous calendar month** (e.g., if today is 2026-01-23, "last month" = 2025-12-01 to 2025-12-31)
- **"last year"** → calculate date range for the previous calendar year (e.g., if today is 2026-01-23, "last year" = 2025-01-01 to 2025-12-31)
- **Specific dates or date ranges** → extract and convert to timestamp format

```python
# Example: Filter for documents after a specific date
metadata_filter = {
    "timestamp": {
        "$gte": start_timestamp,  # Greater than or equal (optional)
        "$lte": end_timestamp      # Less than or equal (optional)
    }
}

# Example: Last month (previous calendar month)
from datetime import datetime, timedelta

today = datetime.now()
# Get first day of current month
first_day_current = today.replace(day=1)
# Get last day of previous month (day before first day of current month)
last_day_previous = first_day_current - timedelta(days=1)
# Get first day of previous month
first_day_previous = last_day_previous.replace(day=1)

start_timestamp = first_day_previous.timestamp()  # Start of last month (e.g., 2025-12-01 00:00:00)
end_timestamp = last_day_previous.timestamp()     # End of last month (e.g., 2025-12-31 23:59:59)

metadata_filter = {
    "timestamp": {
        "$gte": start_timestamp,
        "$lte": end_timestamp
    }
}

# Example: Recent (last 30 days from today)
recent_timestamp = (datetime.now() - timedelta(days=30)).timestamp()
metadata_filter = {
    "timestamp": {"$gte": recent_timestamp}
}
```

**2. String Field Filtering**

For string fields (subject, author, channel_id, etc.):
```python
# Exact match
metadata_filter = {
    "author": "john@example.com"
}

# Multiple conditions
metadata_filter = {
    "channel_id": "C123456",
    "team_id": "T789012"
}
```

**3. Numeric/Boolean Field Filtering**

For numeric fields (group_size) or boolean fields (is_grouped):
```python
# Numeric comparison
metadata_filter = {
    "group_size": {"$gte": 5}  # Groups with 5 or more messages
}

# Boolean
metadata_filter = {
    "is_grouped": True
}
```

**4. Combined Filters**

Combine multiple conditions:
```python
metadata_filter = {
    "$and": [
        {"timestamp": {"$gte": start_timestamp}},
        {"channel_id": "C123456"},
        {"is_grouped": True}
    ]
}
```

**5. No Filter**

If no filter is needed, set `metadata_filter = None`

#### Filter Operators:

- `$gte`: Greater than or equal
- `$lte`: Less than or equal
- `$gt`: Greater than
- `$lt`: Less than
- `$eq`: Equal (default for string fields)
- `$ne`: Matches vectors with metadata values that are not equal to a specified value.
- `$in`: Value in array
- `$nin`: Value not in array
- `$exists`: Matches vectors with the specified metadata field
- `$and`: Joins query clauses with a logical AND
- `$or`: Joins query clauses with a logical OR

#### Examples by Namespace:

**Mailing List:**
```python
# Recent emails (last 30 days from today)
from datetime import datetime, timedelta
recent_timestamp = (datetime.now() - timedelta(days=30)).timestamp()
metadata_filter = {
    "timestamp": {"$gte": recent_timestamp}
}

# Specific thread
metadata_filter = {
    "thread_id": "thread/ABC123"
}
```

**Slack:**
```python
# Recent messages in specific channel (last 7 days)
from datetime import datetime, timedelta
recent_timestamp = (datetime.now() - timedelta(days=7)).timestamp()
metadata_filter = {
    "timestamp": {"$gte": recent_timestamp},
    "channel_id": "C123456"
}

# Thread messages only
metadata_filter = {
    "thread_ts": {"$ne": ""}  # Not empty
}
```

**WG21 Papers:**
```python
# Recent papers (after a specific date)
from datetime import datetime
timestamp_2024 = datetime(2024, 1, 1).timestamp()
metadata_filter = {
    "timestamp": {"$gte": timestamp_2024}
}

# Specific document number
metadata_filter = {
    "document_number": "P0843R10"
}
```

**Documentation:**
```python
# Specific source/library
metadata_filter = {
    "library": "cppreference.com"  # Filter by source: "cppreference.com", "isocpp.github.io", "gcc.gnu.org", "cplusplus.com", "git_MicrosoftDocs", "git_cplusplus"
}

# Multiple sources using $in
metadata_filter = {
    "library": {"$in": ["cppreference.com", "isocpp.github.io"]}
}
```

### 3. Run query.py to Retrieve Results

### 3.1 Initialize PineconeQuery

```python
from query import PineconeQuery

# PineconeQuery loads configuration from environment variables automatically
query_client = PineconeQuery()
```

### 3.2 Execute Query

```python
documents = query_client.query(
    query=query_text,
    top_k=top_k,
    namespace=namespace,
    metadata_filter=metadata_filter,
    use_reranking=True  # Use reranking for better results
)
```

### 4. Post-Processing

### 4.1 Generate Reference URLs from Metadata

**Important**: Only generate URLs for **essential documents** (typically 10-20) that contain the most relevant content for the user's question. Do not generate URLs for all retrieved documents.

For each essential document, generate a reference URL based on the namespace and metadata:

#### Namespace: "mailing"

- Extract `doc_id` or `thread_id` from metadata
- URL format: `https://lists.boost.org/archives/list/{doc_id}/`
- Example: `https://lists.boost.org/archives/list/boost-announce@lists.boost.org/message/O5VYCDZADVDHK5Z5LAYJBHMDOAFQL7P6/`

#### Namespace: "slack-Cpplang"

- Extract `team_id`, `channel_id` and `doc_id` from metadata
- Extract message_id from message_id = doc_id.replace('.', '')
- URL format: `https://app.slack.com/{team_id}/{channel_id}/p{message_id}`
- Alternative format (if available): Use `source` field from metadata directly
- Example: `https://app.slack.com/client/T123456789/C123456/p1234567890`

#### Namespace: "wg21-papers"

- Extract `url` or `doc_id` from metadata
- If `url` exists in metadata, use it directly
- Otherwise, construct from `doc_id` (which has format `wg21/{document_number}` or `wg21/{filename}`):
  - Extract document number from `doc_id` by removing `wg21/` prefix
  - Format: `https://www.open-std.org/jtc1/sc22/wg21/docs/papers/{year}/{document_number}.pdf` or use `url` field if available
- Example: If `doc_id = "wg21/P0999R0"`, extract `P0999R0` and construct URL like `https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0999r0.pdf`

#### Namespace: "cpp-documentation"

- Extract `doc_id` or `url` from metadata (if available)
- Example: `https://www.boost.org/doc/libs/1.89.0/libs/filesystem/doc/index.html`

### 4.2 Generate Systematic Report with Summarization

#### Report Structure:

The answer should be a **logical, systematic report** that synthesizes and summarizes retrieved information, not just a list of document contents. Follow this structure:

1. **Executive Summary** (2-3 sentences)

   - Brief overview of findings
   - Main themes or topics identified
   - Overall conclusion or key takeaway

2. **Main Content** (organized logically)

   - **Group related information** by theme, topic, or chronology
   - **Include rich, detailed content** from retrieved documents: specific library names, feature names, version numbers, API changes, code examples, technical details, and concrete examples
   - **Balance synthesis with detail**: Synthesize information while preserving important specifics from source documents
   - **Include specific examples**: When documents mention specific libraries, features, or changes, include their names and details
   - **Quote key passages**: When a document contains particularly important or informative content, include relevant excerpts (2-4 sentences) with proper citations
   - **Provide concrete details**: Include version numbers, library names, feature descriptions, API changes, bug fix descriptions, and other specific technical information
   - Use subsections if multiple distinct topics are covered
   - **Cite sources** with reference numbers [1], [2], etc. when presenting specific information

3. **Key Findings** (if applicable)

   - Bullet points of most important findings
   - Patterns or trends identified across documents
   - Notable differences or conflicts between sources

4. **References**
   - **Include only essential URLs** (typically 10-20) that contain the most relevant content for the user's question
   - **Do NOT list all retrieved documents** - only include URLs that were actually cited or contain essential information
   - Prioritize URLs based on:
     - Relevance to the user's question
     - Information actually used in the answer
     - Documents with highest similarity scores
     - Unique perspectives or important details
   - Include document metadata (subject, author, date) when available
   - Format: `[N]: {URL} - {Subject/Title} (if available)`

#### Summarization Guidelines:

**DO:**

- **Include rich, detailed content**: Extract and include specific information from documents such as:
  - Library names (e.g., "Boost.Asio", "Boost.Beast", "Boost.Hash2")
  - Feature names and descriptions
  - Version numbers and release dates
  - API changes and new functions/classes
  - Bug fix descriptions
  - Code examples or snippets when available
  - Technical specifications and details
  - Specific examples and use cases
- **Quote informative passages**: Include 2-4 sentence excerpts from documents when they contain valuable technical details, specific examples, or important information
- **Synthesize with specifics**: Combine information from multiple documents while preserving concrete details like library names, feature names, and technical specifics
- **Group** related information by theme or topic, but include the specific details within each group
- **Identify patterns** across documents (e.g., "Multiple discussions [1][2][3] mention that Boost.Asio received new async features...")
- **Create logical flow** between paragraphs and sections
- **Use citations** to support each claim: [1], [2], [3], etc.
- **Prioritize** most relevant information first, but include enough detail to be informative
- **Select essential references only** (10-20 URLs) - do not list all retrieved documents

**DON'T:**

- Don't just list documents one by one unless the user requires it.
- Don't quote excessively long passages (keep excerpts to 2-4 sentences)
- Don't repeat the same information multiple times
- Don't include irrelevant details
- Don't create disconnected paragraphs
- **Don't be too abstract**: Avoid vague statements like "libraries received updates" - instead say "Boost.Asio received new async features X and Y [1]"
- **Don't list all retrieved URLs** - only include essential references (10-20) that are actually cited or contain critical information

#### Answer Format Template:

```markdown
## Answer

### Executive Summary

[2-3 sentence overview of findings and main themes, including specific examples]

### Main Content

#### [Topic/Theme 1]

[Rich, detailed content with specific library names, features, version numbers, etc. Include 2-4 sentence quotes when valuable, with citations [1][2]]

For example: "Boost 1.89 introduces two new libraries: Boost.Hash2 and Boost.MQTT5. Boost.Hash2 provides an extensible hashing framework, while Boost.MQTT5 offers an MQTT5 client library built on top of Boost.Asio [1]. The release announcement states: 'These open-source libraries work well with the C++ Standard Library, and are usable across a broad spectrum of applications' [1]."

[Additional related information with specific details from other documents [3][4]]

#### [Topic/Theme 2] (if applicable)

[Detailed information organized by theme, including specific examples, library names, feature descriptions, with citations [5][6]]

### Key Findings

- Finding 1 with specific details [1][2] (e.g., "Boost.Asio received new async features X and Y [1][2]")
- Finding 2 with concrete examples [3][4]
- Finding 3 with specific information [5]

### References

[1]: {URL_1} - {Subject/Title if available}
[2]: {URL_2} - {Subject/Title if available}
[3]: {URL_3} - {Subject/Title if available}
...
[10]: {URL_10} - {Subject/Title if available}

_Note: Only essential references (10-20) are included. Not all retrieved documents are listed._
```

#### Example of Good Summarization:

**BAD (just listing documents):**

> According to document [1], "Boost.Asio performance can be improved by using async operations." According to document [2], "Memory management is important in Boost.Asio." According to document [3], "Boost.Asio supports various protocols."

**BAD (too abstract, lacks detail):**

> Multiple discussions [1][2][3] highlight several approaches to optimize Boost.Asio applications. Performance improvements can be achieved through proper use of async operations [1], while careful memory management is essential for scalability [2]. The library's support for various network protocols [3] provides flexibility in implementation.

**GOOD (synthesized with rich, specific content):**

> Multiple discussions [1][2][3] highlight several approaches to optimize Boost.Asio applications. Performance improvements can be achieved through proper use of async operations, particularly the new `async_read_some()` and `async_write_some()` functions introduced in Boost 1.89 [1]. One discussion notes: "The new async operations provide better memory efficiency and reduce context switching overhead" [1]. Careful memory management using `boost::asio::buffer` and proper lifetime management of async operation handlers is essential for scalability [2]. The library's support for TCP, UDP, and SSL/TLS protocols [3] provides flexibility in implementation, with recent additions supporting HTTP/2 and WebSocket protocols built on top of Boost.Beast [3].

### 5. Cleanup temporary files

After generating the report, remove all temporary files except:

- Configuration files (e.g., `.env`, `config.py`, etc.)
- The generated report file
- Do not remove any files in the `config/` directory or configuration-related files

### 6. Error Handling

- **Import Errors**: If `PineconeQuery` cannot be imported, check dependencies and install missing packages
- **Connection Errors**: If Pinecone connection fails, check API key and environment settings
- **Empty Results**: Inform user and suggest alternative queries or namespaces
- **Metadata Errors**: If URL generation fails, include document ID or metadata in reference instead

---

## Rules and Guidelines

### 7. Example Workflow

**User Prompt**: "What are the recent discussions about Boost.Asio performance?"

**Extracted Parameters**:

- Query: "Boost.Asio performance"
- top_k: 10 (default)
- namespace: "mailing" (discussions)
- metadata_filter: `{"timestamp": {"$gte": recent_timestamp}}` (recent = last 30 days from today)

**User Prompt**: "What messages were created last month?"

**Extracted Parameters**:

- Query: "message email discussion"
- top_k: 10 (default)
- namespace: "mailing" (or "slack-Cpplang" depending on context)
- metadata_filter: Calculate previous calendar month using the method shown in section 2.4 (e.g., if today is 2026-01-23, last month = 2025-12-01 to 2025-12-31)

**Execution**:

1. Initialize `PineconeQuery`
2. Execute query with extracted parameters (hybrid search with reranking)
3. Retrieve documents from Pinecone
4. Analyze and group documents by themes/topics (e.g., "Performance Optimization", "Memory Management", "Protocol Support")
5. Extract rich, detailed content from documents: library names, feature names, version numbers, API changes, technical details, and concrete examples
6. Summarize and synthesize information from grouped documents while preserving and including specific details
7. Include 2-4 sentence quotes from documents when they contain valuable technical information or examples
8. Identify essential documents (10-20) that contain the most relevant content
9. Generate URLs for essential documents based on namespace
10. Create systematic report with:

- Executive Summary (with specific examples)
- Main Content (organized by themes, with rich details and specific information)
- Key Findings (with concrete examples and specifics)
- References (only essential URLs, 10-20, with metadata)

11. **Cleanup**: Follow the cleanup instructions in section 5

### 8. Technical Notes

- The `PineconeQuery` class uses hybrid search (dense + sparse) with reranking
- Results are automatically deduplicated and sorted by relevance
- Metadata filters support timestamp, string, numeric, boolean, and combined filtering (see section 2.4)
- URL generation rules are namespace-specific and may be updated
- Always use reranking (`use_reranking=True`) for best results unless explicitly requested otherwise
