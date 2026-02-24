# Pinecone Usage Analysis Report

**Author**: Jonathan  
**Org**: CPPAlliance Org  
**Analysis Period**: 2026-01-07 to 2026-02-21 (46 days)  
**Report Date**: 2026-02-21

---

## Purpose

I reviewed Pinecone billing and usage data to identify practical optimization actions I can take now without harming retrieval quality.

---

## Current Architecture

I currently run two serverless indexes in `gcp/us-central1`:

- `rag-hybrid` (dense): `llama-text-embed-v2`
- `rag-hybrid-sparse` (sparse): `pinecone-sparse-english-v0`

I need this dual-index setup because I sometimes run sparse-only retrieval.

Current embed runtime settings confirmed from live indexes:

- `rag-hybrid`: write/read `truncate=END`
- `rag-hybrid-sparse`: write/read `truncate=END`
- `rag-hybrid-sparse`: `max_tokens_per_sequence=512`

---

## Index stats (current)

### Dense index (`rag-hybrid`)

| Metric         |     Value |
| -------------- | --------: |
| Total vectors  | 6,154,037 |
| Dimension      |     1,024 |
| Index fullness |       0.0 |

### Sparse index (`rag-hybrid-sparse`)

| Metric         |      Value |
| -------------- | ---------: |
| Total vectors  |  6,151,853 |
| Dimension      | 0 (sparse) |
| Index fullness |        0.0 |

**Vectors by namespace (Dense vs Sparse, delta = Dense − Sparse):**

| Namespace         |     Dense |    Sparse |  Delta |
| ----------------- | --------: | --------: | -----: |
| blog-posts        |    30,207 |    30,207 |      0 |
| cpp-documentation |   353,664 |   353,659 |     +5 |
| github-clang      | 2,639,888 | 2,639,792 |    +96 |
| mailing           |   680,628 |   680,628 |      0 |
| slack-Cpplang     |   387,723 |   387,560 |   +163 |
| wg21-papers       | 1,470,982 | 1,470,982 |      0 |
| youtube-scripts   |   590,945 |   589,025 | +1,920 |

#### Namespaces with dense/sparse mismatch

The following namespaces have different vector counts in the dense and sparse indexes. A common cause is the **sparse embedding model**: the same text is upserted to both indexes, but the sparse model can sometimes return vectors with **very few or zero non-zero elements**. Pinecone’s API requires sparse vectors to have at least one element in `indices`/`values` ([Upsert vectors](https://docs.pinecone.io/reference/api/data-plane/upsert) — `SparseValues` has `minLength: 1`). Records with empty or near-empty sparse embeddings can therefore be stored in the dense index but fail or be skipped for the sparse index, producing a positive delta (dense > sparse). Other causes include partial runs or failures on one side.

| Namespace         |     Dense |    Sparse |  Delta |
| ----------------- | --------: | --------: | -----: |
| youtube-scripts   |   590,945 |   589,025 | +1,920 |
| github-clang      | 2,639,888 | 2,639,792 |    +96 |
| slack-Cpplang     |   387,723 |   387,560 |   +163 |
| cpp-documentation |   353,664 |   353,659 |     +5 |

See [Empty Sparse Vector](https://community.pinecone.io/t/empty-sparse-vector/5151) and [Upsert vectors](https://docs.pinecone.io/reference/api/data-plane/upsert).

---

## Cost Summary

_Share = each category’s percentage of total cost._

| Category                          |         Cost | Share of total |
| --------------------------------- | -----------: | -------------: |
| Inference (embedding + rerank)    |  **$526.18** |      **65.6%** |
| Database (write + storage + read) |  **$275.79** |      **34.4%** |
| **Total**                         | **~$801.97** |       **100%** |

### Inference breakdown

| Model                        |         Usage |    Cost |
| ---------------------------- | ------------: | ------: |
| `llama-text-embed-v2`        | ~2.72B tokens | $327.30 |
| `pinecone-sparse-english-v0` | ~3.34B tokens | $198.55 |
| `bge-reranker-v2-m3`         |  146 requests |   $0.33 |

### Database breakdown

| Item        | Index               |    Cost |
| ----------- | ------------------- | ------: |
| Write Units | `rag-hybrid`        | $199.34 |
| Write Units | `rag-hybrid-sparse` |  $67.54 |
| Storage     | both                |   $8.85 |
| Read Units  | both                |   $0.06 |

---

## Cost Pattern I Observed

- The largest spike is **Feb 20 (~$204.64/day)**.
- Idle days are mostly storage-only spend.
- Read costs are negligible relative to ingestion costs.

Representative ingestion days:

| Date       | Dense embed | Sparse embed | Dense write | Sparse write |    Day total |
| ---------- | ----------: | -----------: | ----------: | -----------: | -----------: |
| Jan 12     |      $39.78 |       $23.62 |      $28.42 |       $10.09 |     ~$101.91 |
| Jan 23     |      $28.47 |       $17.16 |      $21.02 |        $6.61 |      ~$73.26 |
| Feb 5      |      $49.56 |       $30.04 |      $30.66 |       $10.02 |     ~$120.28 |
| Feb 19     |      $25.41 |       $15.96 |      $13.04 |        $4.36 |      ~$58.77 |
| **Feb 20** |  **$91.20** |   **$56.13** |  **$43.17** |   **$14.14** | **~$204.64** |

---

## Key Findings

1. My primary cost driver is **embedding inference** (65.7%).
2. Within database spend, **write units dominate**.
3. Current chunking (`chunk_size=1000`) likely increases record count and write/embedding volume.
4. The dual-index design adds overhead, but it is justified for sparse-only search needs.

---

## Optimization Plan (Priority Order)

### 1) Increase `chunk_size` first (highest practical impact)

**Why this is my top recommendation**  
Given my architecture constraints, reducing chunk count is the strongest and safest lever.

**Action**

```python
# config.py
chunk_size = 2000
chunk_overlap = 300
```

**Expected impact**

- Fewer chunks
- Lower embedding tokens
- Lower write units
- Practical savings typically in the 20–40% range (dataset-dependent)

**Safety note**  
`2000` chars is usually close to ~500 tokens, which is near but generally within sparse `2048` max tokens(default 512).

Reference: [sparse-model](https://docs.pinecone.io/models/llama-text-embed-v2)

---

### 2) Avoid redundant re-embedding and writes (two levers)

**Same goal, different situations:**

- **Skip-existing-ID (upsert guard):** I implemented skip-existing-ID logic in `ingestion.py`. When upserting, if the record ID already exists, we skip it—no re-embed, no re-upsert. This has limited impact if I do not rerun the same data, but it protects against accidental duplicate ingestion and recovery reruns.
- **Metadata update API (metadata-only changes):** When only metadata changes (e.g. labels, status, source URL), use Pinecone’s **update** API instead of re-ingesting: no re-embedding, no full upsert. See §4 below for technique, when to apply it, and cost evidence.

Reference: [Upsert records](https://docs.pinecone.io/guides/index-data/upsert-data)

---

### 3) Use metadata filtering for relevance quality

Because read cost is already tiny, metadata filtering is mostly a quality optimization, not a major cost optimization.

Reference: [Filter by metadata](https://docs.pinecone.io/guides/search/filter-by-metadata)

---

### 4) Use metadata update API for metadata-only changes (write-time)

**Technique**  
Use Pinecone’s **metadata update** API instead of re-ingesting vectors when only metadata changes. Do not re-embed or re-upsert the same content.

**When to apply**  
When a document’s **metadata** changes (e.g. labels, status, timestamps, source URL) but the **content** is unchanged. If content changed, a full re-embed and upsert is still required.

**Cost/IO benefit**

- Avoids **re-embedding** (no extra embedding inference or tokens).
- Avoids **full write** of the vector; only metadata is updated.
- Reduces write-unit and embedding cost on metadata-only refresh or correction runs.

**Evidence (Pinecone docs)**

- **Embedding:** [Understanding cost](https://docs.pinecone.io/guides/manage-cost/understanding-cost) states that embedding cost is per token. Metadata-only update does not call the embedding API, so **embedding cost is zero** for those operations. In this setup, inference (embedding + rerank) is the main cost (e.g. 65.7%), so skipping re-embedding is where most of the saving comes from.
- **Write units:** The same guide states that [Update](https://docs.pinecone.io/guides/manage-cost/understanding-cost#update) uses 1 WU per 1 KB of the new and existing record (min 5 WUs per request). For metadata-only updates the request payload is small (no vector), so write-unit cost per record is lower than a full upsert of the same record.
- **Update behavior:** [Update records](https://docs.pinecone.io/guides/manage-data/update-data) confirms that when updating metadata, only the specified metadata fields are modified; vector values are unchanged, so no re-embedding or full-record overwrite is needed.

Reference: [Update vectors](https://docs.pinecone.io/guides/manage-data/update-data)

---

## Final Action Items

1. Change chunking to `2000/300` and validate retrieval quality on a sample query set.
2. Keep skip-existing-ID guard enabled.
3. Use metadata `update` for metadata-only changes.
