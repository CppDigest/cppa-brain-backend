# MailHierarchicalRAG – Retrieval and Recency Weighting Report

## Overview
`MailHierarchicalRAG` builds and searches an email-thread knowledge graph. It supports:
- Semantic vector search (FAISS, cosine similarity)
- Keyword (BM25-like) text search over synthesized node text
- Graph-based search (node + neighbor embedding similarity)
- A hybrid pipeline that stages these methods and applies a recency-aware reweighting

This document explains how ranking works across semantic, keyword and graph search, and how recency weighting is applied uniformly.

## Data and Graph
- Nodes: email messages and threads
- Edges: parent/child reply relationships
- Stored attributes per node include: `message_id`, `subject`, `sender_address`, `date`, `to`, `cc`, `reply_to`, `url`, `summary`.
- A FAISS inner-product index (cosine with L2 normalization) is built from per-node embeddings for fast semantic retrieval.

## Synthesized Node Text
For searchable text, each node is represented by a short synthesized text composed from node metadata:
- subject: …
- summary: …
- from: sender_address

This is used in keyword retrieval and for display.

## Search Methods
### 1) Semantic Search
- Query is embedded, normalized, and searched against the FAISS index.
- Early filtering: a dynamic threshold is set at `0.8 * top_score` (from the FAISS result set) to discard low-similarity hits fast.
- Recency weighting is then applied to the retained results (details below). The method writes a `gen_score` for downstream use.

### 2) Keyword Search
- Computes a simple match score from the synthesized node text (term presence + small bonus for exact-phrase match).
- Thresholding: keeps results with `score >= 0.8 * top_score`.
- Recency weighting: applied to the filtered set; `gen_score` is updated.

### 3) Graph Search (Embedding Synthesis)
- Builds a query embedding and computes cosine similarity with:
  - The current node’s normalized embedding
  - All immediate neighbors (predecessors and successors), then aggregates as:
    - `0.6 * sim(node) + 0.4 * mean(sim(neighbors))`
- Thresholding: keeps results with `score >= 0.7 * top_score`.
- Recency weighting: applied to the filtered set; `gen_score` is updated.

## Recency Weighting (Uniform Across Methods)
A common helper `_apply_recency` adjusts scores using email dates from node metadata:
- Recency weight: `exp(-age_days / half_life_days)`
  - Config: `rag.retrieval.mail.recency_half_life_days` (default: 1800)
  - Clamp: `weight ∈ [0.1, 1.0]` to avoid zeroing older but relevant items
- Final contribution (written to `gen_score`):
  - `gen_score = base * prev_scale + raw_score * (alpha + beta * recency_weight) * add_scale`
  - Default blend per stage:
    - Semantic: `alpha=0.7`, `beta=0.3`, no accumulation
    - Keyword: threshold then `alpha=0.7`, `beta=0.3`, no accumulation (downstream aggregation uses the `gen_score` later)
    - Graph: threshold then `alpha=0.7`, `beta=0.3`, no accumulation

This produces a balanced signal where newer content (higher recency weight) is favored, without overwhelming relevance.

## Thresholding and Tail Processing
A shared `_finalize_results` helper performs:
- Optional thresholding by ratio of the top score
- Recency reweighting (writes `gen_score`)
- Optional sort by `gen_score` (fallback to `score`)
- Optional slicing by desired `limit`

Default threshold ratios:
- Semantic: early cut via FAISS top score (0.8×) before result construction
- Keyword: 0.8× top score
- Graph: 0.7× top score

## Hybrid Flow
The hybrid pipeline composes methods sequentially:
1) Semantic search over the filtered node set → produce `semantic_results`
2) Restrict to node IDs returned by semantic; run Keyword search → `keyword_results`
3) Restrict to node IDs returned by keyword; run Graph search → `graph_results`
4) Sort `graph_results` by `gen_score` and return the top `N` (paging happens higher in the orchestrator)

This flow lets the more expensive graph scoring run on a narrowed set, while recency-aware reweighting is applied consistently at each stage.

## Configuration Summary
- FAISS backend: inner-product with normalized vectors (cosine)
- Recency:
  - Half-life: `rag.retrieval.mail.recency_half_life_days` (default 1800)
  - Clamp: `[0.1, 1.0]`
- Threshold ratios:
  - Semantic: ~0.8× (via FAISS head threshold)
  - Keyword: 0.8×
  - Graph: 0.7×

## Key Helpers
- `_threshold_by_top(results, ratio)` – keep results above ratio × top_score
- `_apply_recency(results, alpha, beta, accumulate, prev_scale, add_scale)` – applies date-based reweighting into `gen_score`
- `_normalize_vector(vec)` / `_get_normalized_node_vector(node_id)` – common cosine normalization
- `_finalize_results(…)` – shared tail processing (threshold, recency, sort, slice)

## Notes & Rationale
- Recency favors timely, actively discussed items without eliminating high-relevance older content, thanks to clamping and blended weighting.
- Staged hybrid minimizes cost for graph scoring and allows each method to prune and reweight results consistently.
- Aggregating node and neighbor similarities in graph search helps surface messages that are central to relevant sub-conversations.
