# Pinecone RAG database schema

**Schema version:** 1.0 · **Last updated:** 2026-02-26

Index layout, namespaces, and document shape (content + metadata) for data upserted by the cloud_rag pipeline. Reference for upserts, MCP tools, and query filters.

---

## 1. Index topology

Two serverless indexes (integrated embedding):

| Index name          | Type   | Embedding model              | Metric     |
| ------------------- | ------ | ---------------------------- | ---------- |
| `rag-hybrid`        | Dense  | `llama-text-embed-v2`        | cosine     |
| `rag-hybrid-sparse` | Sparse | `pinecone-sparse-english-v0` | dotproduct |

- **Field map:** `text` → `chunk_text` (required on every record).
- **Chunking:** Default `chunk_size=2000`, `chunk_overlap=200` unless pre-chunked.

---

## 2. Record shape

One record = one chunk; same `id` and metadata in both indexes.

| Field        | Type   | Description                                                                                      |
| ------------ | ------ | ------------------------------------------------------------------------------------------------ |
| `id`         | string | Unique in namespace. MD5 of `(doc_id or url)` + chunk suffix (`start_index` or `text[:50]_len`). |
| `chunk_text` | string | Text to embed (required). Ingestion may prepend `Title: {metadata.title}\n\n` when present.      |

**Reserved** (not stored as metadata): `id`, `chunk_text`. All other document metadata is stored as record metadata (Pinecone types: string, number, boolean, list of strings).

---

## 3. Namespaces and data sources

| Namespace           | Purpose                                                                    | Source(s)                           | Preprocessor(s)                                                                          |
| ------------------- | -------------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------- |
| `mailing`           | Mailing list threads                                                       | `mail`                              | MailPreprocessor                                                                         |
| `cpp-documentation` | C++/Boost docs (markdown)                                                  | `doc`                               | DocuPreprocessor                                                                         |
| `slack-Cpplang`     | Slack messages                                                             | `slack`                             | SlackPreprocessor                                                                        |
| `wg21-papers`       | WG21 C++ papers                                                            | `wg21_papers`                       | WG21PaperPreprocessor                                                                    |
| `youtube-scripts`   | YouTube transcripts (by segment)                                           | `youtube`                           | YouTubePreprocessor                                                                      |
| `blog-posts`        | Blog JSON + PDFs (default namespace for both BlogConfig and BlogPdfConfig) | `blog_posts`, `stroustrup_pdf`      | BlogPreprocessor, PdfPreprocessor                                                        |
| `github-clang`      | GitHub issues/PRs, Phabricator diffs, Bugzilla issues (filter by `type`)   | `github`, `phabricator`, `bugzilla` | GitIssuePreprocessor, GitPrPreprocessor, PhabricatorPrPreprocessor, BugIssuePreprocessor |

Namespace values come from `config.py` (env: `*_NAMESPACE`). PdfPreprocessor defaults to `blog-posts`; override with `BLOG_PDF_NAMESPACE` if needed.

---

## 4. Metadata by namespace

Content = `chunk_text`. Metadata below is stored per record (see §5 for `id` generation).

### 4.1 `mailing`

| Field     | Type   | Description       |
| --------- | ------ | ----------------- |
| doc_id    | string | Message ID        |
| type      | string | `"mailing"`       |
| thread_id | string | Thread identifier |
| subject   | string | Subject line      |
| author    | string | Sender address    |
| timestamp | float  | Unix timestamp    |
| parent_id | string | Parent message ID |

### 4.2 `cpp-documentation`

| Field      | Type   | Description                   |
| ---------- | ------ | ----------------------------- |
| doc_id     | string | Source URL                    |
| lang       | string | `"en"`                        |
| type       | string | `"documentation"`             |
| library    | string | From path (e.g. library name) |
| version    | string |                               |
| build_time | float  | Build timestamp               |

### 4.3 `slack-Cpplang`

| Field      | Type   | Description                   |
| ---------- | ------ | ----------------------------- |
| doc_id     | string | Message `ts`                  |
| team_id    | string | Slack team ID                 |
| type       | string | `"slack"`                     |
| channel_id | string | Channel ID                    |
| user_name  | string | Display/username              |
| timestamp  | int    | Message ts                    |
| is_grouped | bool   | Whether messages were grouped |
| thread_ts  | string | Thread ts if any              |
| group_size | int    | Number of messages in group   |

### 4.4 `wg21-papers`

| Field           | Type      | Description          |
| --------------- | --------- | -------------------- |
| document_number | string    | Paper number         |
| type            | string    | `wg21-papers`        |
| title           | string    | Paper title          |
| author          | list[str] | List of author names |
| timestamp       | float     | From paper date      |
| url             | string    | Paper URL            |

### 4.5 `youtube-scripts`

Pre-chunked (one record per segment).

| Field            | Type   | Description                    |
| ---------------- | ------ | ------------------------------ |
| doc_id           | string | `{video_id}_{segment_index}`   |
| type             | string | `"youtube_video"`              |
| video_id         | string | YouTube video ID               |
| url              | string | Watch URL with `t=` start time |
| channel_title    | string | Channel name                   |
| published_at     | float  | Publish timestamp              |
| duration_seconds | int    | Video duration                 |
| view_count       | int    | View count                     |
| like_count       | int    | Like count                     |
| search_term      | string | Search term used to find video |
| start_time       | float  | Segment start (seconds)        |
| end_time         | float  | Segment end (seconds)          |

### 4.6 `blog-posts`

**blog-html (JSON):**

| Field     | Type   | Description       |
| --------- | ------ | ----------------- |
| title     | string | Post title        |
| url       | string | Post URL          |
| author    | string | Author            |
| timestamp | float  | Publish timestamp |
| type      | string | `"blog-html"`     |

**blog-pdf (PDF):**

| Field       | Type   | Description          |
| ----------- | ------ | -------------------- |
| doc_id      | string | `{url}#p{page}`      |
| type        | string | `"blog-pdf"`         |
| author      | string | From config          |
| title       | string | From PDF or filename |
| url         | string | Source URL           |
| total_pages | int    | Page count           |
| timestamp   | float  | From PDF metadata    |

### 4.7 `github-clang`

Filter by `type` to distinguish source. Types: `issue`, `pr`, `pr-phabricator`, `issue-bugzilla`.

**Issues (GitIssuePreprocessor):**

| Field        | Type   | Description    |
| ------------ | ------ | -------------- |
| author       | string | GitHub login   |
| title        | string | Issue title    |
| number       | int    | Issue number   |
| url          | string | HTML URL       |
| created_at   | float  | Unix timestamp |
| updated_at   | float  | Unix timestamp |
| closed_at    | float  | Unix timestamp |
| type         | string | `"issue"`      |
| state        | string | open / closed  |
| state_reason | string |                |

Issues: LLVM GitHub org only.

**PRs:**

| Field      | Type   | Description    |
| ---------- | ------ | -------------- |
| type       | string | `"pr"`         |
| number     | int    | PR number      |
| title      | string | PR title       |
| url        | string | HTML URL       |
| author     | string | GitHub login   |
| state      | string | open / closed  |
| created_at | float  | Unix timestamp |
| updated_at | float  | Unix timestamp |
| closed_at  | float  | Unix timestamp |

**Phabricator:**

| Field        | Type   | Description                        |
| ------------ | ------ | ---------------------------------- |
| type         | string | `"pr-phabricator"`                 |
| number       | int    | Differential number (D123)         |
| title        | string | Diff title                         |
| url          | string | e.g. https://reviews.llvm.org/D123 |
| author       | string | Username                           |
| state        | string | open / closed / abandoned / merged |
| state_reason | string |                                    |
| created_at   | float  | Unix timestamp                     |
| updated_at   | float  | Last activity timestamp            |
| closed_at    | float  | Unix timestamp (if closed)         |

**Bugzilla:**

| Field        | Type   | Description                                    |
| ------------ | ------ | ---------------------------------------------- |
| type         | string | `"issue-bugzilla"`                             |
| number       | int    | Bug ID                                         |
| title        | string | Bug summary                                    |
| url          | string | e.g. https://bugs.llvm.org/show_bug.cgi?id=123 |
| author       | string | Creator                                        |
| state        | string | Bug status                                     |
| state_reason | string | Resolution                                     |
| created_at   | float  | Unix timestamp                                 |
| updated_at   | float  | Last change timestamp                          |
| closed_at    | float  | Unix timestamp (0 if open)                     |

---

## 5. ID generation

`original_doc_id = metadata.get("doc_id", metadata.get("url"))`; then append `_start_index` or `_text[:50]_len(text)`. `id = md5(original_doc_id)`.

---

## 6. Query and filtering

Query one namespace; use dense + sparse indexes and optional rerank. Filter by any stored metadata (Pinecone filter syntax).
