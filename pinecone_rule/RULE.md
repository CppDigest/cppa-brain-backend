---
description: "Rules for querying Pinecone using hybrid search and generating systematic, synthesized reports with logical organization and proper citations"
alwaysApply: false
---

# Pinecone Query and Answer Generation Rule

When asked to retrieve information or answer questions using Pinecone vector database:

## 1. Environment Setup

**Check and install required dependencies:**

- Check if `python-dotenv>=1.0.0` is installed
- Check if `pinecone>=3.0.0` is installed
- Check if `python-dotenv>=1.0.0` is installed

- If not installed, run: `pip install python-dotenv>=1.0.0 pinecone>=3.0.0 python-dotenv>=1.0.0`
- Ensure `.env` file exists with required Pinecone configuration:
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME` (default: "rag-hybrid")
  - `PINECONE_RERANK_MODEL` (default: "bge-reranker-v2-m3")

## 2. Read User's Prompt

- Extract the user's question or information request from their prompt
- Identify the main query intent and any specific requirements

## 3. Extract Parameters for query.py

### 3.1 Query Text

- Extract the core query text from the user's prompt
- Clean and normalize the query (remove unnecessary words, but preserve intent)
- This will be passed as the `query` parameter to `PineconeQuery.query()`

### 3.2 Determine top_k

- Default: `10` documents
- Adjust based on user's request:
  - If user asks for "a few" or "several": use `5`
  - If user asks for "many" or "comprehensive": use `20`
  - If user specifies a number: use that number (within reasonable limits: 1-10000)
- Pass as `top_k` parameter

### 3.3 Determine Namespace

- Analyze user's question to determine the appropriate namespace:
  - **"mailing"**: For questions about Boost mailing list discussions, email threads, community discussions
  - **"slack"**: For questions about Slack conversations, team discussions, chat history
  - **"wg21"**: For questions about C++ standard proposals, WG21 documents
  - **"cpp-documentation"**: For questions about C++ documentation, Boost library documentation
  - **"documentation"**: Alternative name for cpp-documentation
- If namespace cannot be determined from context, default to **"mailing"**
- Pass as `namespace` parameter

### 3.4 Extract Metadata Filter (Timestamp)

- Look for time-related keywords in user's prompt:
  - "recent", "latest", "new", "current" → filter for recent documents
  - "last week", "last month", "last year" → calculate date range
  - Specific dates or date ranges → extract and convert to timestamp format
- Build `metadata_filter` dictionary:
  ```python
  metadata_filter = {
      "timestamp": {
          "$gte": start_timestamp,  # Greater than or equal (optional)
          "$lte": end_timestamp     # Less than or equal (optional)
      }
  }
  ```
- If no time filter is needed, set `metadata_filter = None`
- Note: Currently only timestamp filtering is supported

## 4. Run query.py to Retrieve Results

### 4.1 Initialize PineconeQuery

```python
from query import PineconeQuery
from config import PineconeConfig

# Load configuration from environment
pinecone_config = PineconeConfig()
query_client = PineconeQuery(pinecone_config=pinecone_config)
```

### 4.2 Execute Query

```python
documents = query_client.query(
    query=query_text,
    top_k=top_k,
    namespace=namespace,
    metadata_filter=metadata_filter,
    use_reranking=True  # Use reranking for better results
)
```

### 4.3 Handle Results

- Check if documents were retrieved
- If no documents found, inform the user and suggest:
  - Trying a different query
  - Checking a different namespace
  - Adjusting time filters if applied

## 5. Post-Processing

### 5.1 Generate Reference URLs from Metadata

**Important**: Only generate URLs for **essential documents** (typically 10-20) that contain the most relevant content for the user's question. Do not generate URLs for all retrieved documents.

For each essential document, generate a reference URL based on the namespace and metadata:

#### Namespace: "mailing"

- Extract `doc_id` or `thread_id` from metadata
- URL format: `http://lists.boost.org/archives/list/{doc_id}/`
- Example: `https://lists.boost.org/archives/list/boost-announce@lists.boost.org/message/O5VYCDZADVDHK5Z5LAYJBHMDOAFQL7P6/`

#### Namespace: "slack"

- Extract `channel_id` and `doc_id` from metadata
- Extract `timestamp` from metadata
- URL format: `slack://{team_id}/{channel_id}/{doc_id}`
- Alternative format (if available): Use `source` field from metadata directly
- Example: `slack://team_1/channel_C123456/message_M789012`

#### Namespace: "wg21"

- Extract `url` or `document_id` from metadata
- If `url` exists in metadata, use it directly
- Otherwise, construct from `document_id`:
  - Format: `https://wg21.link/{document_id}` or similar (to be confirmed)
- Example: `https://wg21.link/P1234R5`

#### Namespace: "cpp-documentation" or "documentation"

- Extract `url` from metadata (if available)
- If `url` exists, use it directly
- Otherwise, construct from `source` and `library` fields:
  - Format: `https://www.boost.org/doc/libs/{version}/libs/{library}/...`
- Example: `https://www.boost.org/doc/libs/1.89.0/libs/filesystem/doc/index.html`

**Note**: URL generation rules may be updated later. For now, use the patterns above or extract `url` directly from metadata if available.

### 5.2 Generate Systematic Report with Summarization

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

- Don't just list documents one by one
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

#### Best Practices:

- **Include specific details**: Always include concrete information like library names, feature names, version numbers, API changes, and technical specifications
- **Quote valuable content**: Include 2-4 sentence excerpts when documents contain particularly informative technical details, examples, or important information
- **Balance synthesis with detail**: Synthesize information from multiple sources while preserving and including the specific details from each source
- **Synthesize first, cite second**: Create coherent insights with rich content, then cite sources
- **Group by theme**: Organize information logically rather than by document order, but include specific details within each theme
- **Use multiple citations**: When information appears in multiple sources, cite all: [1][2][3]
- **Prioritize relevance**: Most relevant information should appear first, with sufficient detail to be informative
- **Maintain flow**: Use transitional phrases to connect ideas while preserving specific details
- **Be informative, not just concise**: Include enough detail to be valuable - prefer "Boost.Asio received new async features X and Y [1]" over "Boost.Asio received updates [1]"
- **Check completeness**: Ensure all cited references appear in References section
- **Filter references**: Only include essential URLs (10-20) that are actually cited or contain critical information for the answer

## 6. Error Handling

- **Import Errors**: If `PineconeQuery` cannot be imported, check dependencies and install missing packages
- **Connection Errors**: If Pinecone connection fails, check API key and environment settings
- **Empty Results**: Inform user and suggest alternative queries or namespaces
- **Metadata Errors**: If URL generation fails, include document ID or metadata in reference instead

## 7. Example Workflow

**User Prompt**: "What are the recent discussions about Boost.Asio performance?"

**Extracted Parameters**:

- Query: "Boost.Asio performance"
- top_k: 10 (default)
- namespace: "mailing" (discussions)
- metadata_filter: `{"timestamp": {"$gte": recent_timestamp}}` (recent discussions)

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

11. **Cleanup**: After generating the report, remove all temporary files except:

- Configuration files (e.g., `.env`, `config.py`, etc.)
- The generated report file
- Do not remove any files in the `config/` directory or configuration-related files

## Technical Notes

- The `PineconeQuery` class uses hybrid search (dense + sparse) with reranking
- Results are automatically deduplicated and sorted by relevance
- Metadata filters currently support timestamp filtering only
- URL generation rules are namespace-specific and may be updated
- Always use reranking (`use_reranking=True`) for best results unless explicitly requested otherwise
- **Answer generation should balance synthesis with rich content**: Synthesize information while including specific details, library names, feature names, version numbers, and technical specifications from source documents
- **Include concrete examples**: When documents mention specific libraries, features, or changes, include their names and detailed descriptions
- **Quote valuable passages**: Include 2-4 sentence excerpts when documents contain particularly informative content
- Group retrieved documents by themes/topics before writing the report
- Prioritize creating coherent insights while preserving important specifics from source documents
- The report format emphasizes logical organization and systematic presentation of findings with sufficient detail to be informative and valuable
- **Cleanup**: After report generation, remove all temporary files except configuration files (`.env`, `config.py`, etc.) and the generated report file
