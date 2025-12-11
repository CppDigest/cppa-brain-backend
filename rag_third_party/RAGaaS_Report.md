# RAGaaS Market Report for C++ Copilot

## Executive Summary

This report analyzes RAGaaS platforms for a **C++ Copilot** use case requiring support for code files (C++, C), documentation (HTML, JSON, PDF, MD, TXT, ADOC), hybrid retrieval methods, sophisticated system prompts, and high accuracy for QA and summarization.

---

## C++ Copilot Requirements

- **Data Types:** Code files (cpp, hpp, c, h), HTML, JSON, PDF, Markdown, TXT, AsciiDoc
- **Data Sources:** Documentation, GitHub, HuggingFace, compiler outputs, Slack, mailing lists, Reddit
- **Capabilities:** QA, Summarization, Graph RAG (knowledge graphs for entity relationships), Hybrid retrieval (vector + keyword + semantic), High accuracy, Sophisticated system prompts

---

## Comparison Summary

| Platform             | Code Support | Graph RAG | Hybrid Retrieval | System Prompts | GitHub/Slack | Accuracy   | Pricing           | Recommendation |
| -------------------- | ------------ | --------- | ---------------- | -------------- | ------------ | ---------- | ----------------- | -------------- |
| **LlamaCloud**       | ⭐⭐⭐⭐⭐   | ✅ Yes    | ✅ Yes           | ✅ Advanced    | ✅ Native    | ⭐⭐⭐⭐⭐ | Usage-based       | **BEST FIT**   |
| **Circlemind**       | ⭐⭐⭐⭐     | ✅ Yes    | ✅ Yes           | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐⭐⭐ | $300/mo           | **GRAPH RAG**  |
| **Lettria**          | ⭐⭐⭐⭐     | ✅ Yes    | ✅ Yes           | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐⭐   | €600/mo           | **GRAPH RAG**  |
| **Progress Agentic** | ⭐⭐⭐⭐     | ⚠️ Custom | ✅ Yes           | ✅ Advanced    | ⚠️ Custom    | ⭐⭐⭐⭐   | $700-1,750/mo     | **STRONG**     |
| **Ragie**            | ⭐⭐⭐⭐     | ❌ No     | ✅ Yes           | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐⭐   | $100-500/mo       | **GOOD**       |
| **Vectara**          | ⭐⭐⭐       | ❌ No     | ✅ Yes           | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐⭐⭐ | Free tier + usage | **SOLID**      |
| **Nuclia**           | ⭐⭐⭐       | ❌ No     | ✅ Yes           | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐⭐   | Contact           | **MULTIMODAL** |
| **Graphlit**         | ⭐⭐⭐       | ❌ No     | ⚠️ Limited       | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐     | $49/mo            | **BUDGET**     |
| **Credal**           | ⭐⭐         | ❌ No     | ⚠️ Limited       | ⚠️ Limited     | ✅ Slack     | ⭐⭐⭐     | $500/mo           | **SECURITY**   |
| **CustomGPT**        | ⭐⭐         | ❌ No     | ⚠️ Limited       | ⚠️ Limited     | ❌ No        | ⭐⭐       | $89/mo            | **NOT IDEAL**  |

---

## Recommended Platforms

### 1. LlamaCloud (by LlamaIndex) ⭐ **BEST FIT**

**URL:** https://llamaindex.ai/llamacloud

**Technology:**

- ✅ Native code parsing with AST support
- ✅ Graph RAG support (via NebulaGraph integration, knowledge graphs)
- ✅ Hybrid retrieval (vector, keyword, hybrid search)
- ✅ GitHub integration (native)
- ✅ Sophisticated system prompts
- ✅ Code-aware embeddings, function/class-level chunking

**Pricing:**

- **Model:** Usage-based (contact for pricing)
- **Free tier:** Available for development
- **Charges based on:** Data ingestion (GB indexed), Query/request count, Compute resources, Storage (GB/month)
- **Best for:** High token usage, flexible scaling

**C++ Copilot Suitability: ⭐⭐⭐⭐⭐**

- Native code file support, repository-level context, multi-hop reasoning

---

### 2. Progress Agentic RAG ⭐ **STRONG CANDIDATE**

**URL:** https://www.progress.com/agentic-rag

**Technology:**

- ✅ Modular architecture for custom code handling
- ✅ Hybrid retrieval (modular approach)
- ✅ Supports any LLM (code-specialized models)
- ✅ Enterprise security, hybrid deployment
- ⚠️ Custom integration needed for GitHub/Slack

**Pricing:**

- **Fly Plan:** $700/month
  - 10GB indexed data, 15K resources, 750MB per file
  - 10,000 tokens/month included
  - Additional: $0.008 per token (verify if per token or per 1K tokens)
- **Growth Plan:** $1,750/month (higher limits)
- **Enterprise:** Custom pricing
- **Free Trial:** 14-day
- **Cost Example:** 50K tokens/month = $700 + (40K × $0.008) = ~$1,020/month

**C++ Copilot Suitability: ⭐⭐⭐⭐**

- Modular RAG components, custom LLM integration, enterprise security

---

### 3. Ragie ⭐ **GOOD FIT**

**URL:** https://ragie.ai

**Technology:**

- ✅ Hybrid search (explicitly supported)
- ✅ Multimodal indexing (code + docs)
- ✅ LLM re-ranking
- ✅ Entity extraction (functions, classes)
- ⚠️ Custom connectors needed for GitHub/compiler

**Pricing:**

- **Freemium:** $100/month (limited data/requests)
- **Standard:** $500/month (higher limits, full features, connectors)
- **Model:** Tier-based with overage charges
- **Contact:** For specific limits and overage pricing

**C++ Copilot Suitability: ⭐⭐⭐⭐**

- Hybrid search, summary indexing, bank-grade security

---

### 4. Vectara ⭐ **SOLID OPTION**

**URL:** https://vectara.com

**Technology:**

- ✅ High accuracy with re-ranking
- ✅ Hybrid retrieval with re-ranking
- ✅ End-to-end RAG pipeline
- ✅ Easy API integration
- ⚠️ May need custom code parsing

**Pricing:**

- **Free tier:** Available (limited requests/data)
- **Model:** Usage-based (contact for pricing)
- **Charges based on:** Data indexing (per document/page), Query requests (per query/1K queries), Storage (GB/month), Token usage
- **Best for:** Pay-as-you-go, high accuracy needs

**C++ Copilot Suitability: ⭐⭐⭐⭐**

- Complete ML search process, high accuracy results

---

### 5. Nuclia ⭐ **MULTIMODAL CHOICE**

**URL:** https://nuclia.com

**Technology:**

- ✅ Multimodal support (text, docs, videos, code)
- ✅ AI-driven hybrid search
- ✅ Integration with Progress Agentic RAG
- ⚠️ Less code-specific than LlamaIndex

**Pricing:**

- **Model:** Contact-based (usage-based)
- **Charges based on:** Indexed data volume (GB), Search queries (per 1K), Processing resources, Storage
- **Best for:** Enterprise custom deployments

**C++ Copilot Suitability: ⭐⭐⭐**

- Multimodal indexing, seamless search across formats

---

### 6. Graphlit

**URL:** https://graphlit.com

**Technology:**

- ✅ Multiple LLMs (OpenAI, Anthropic, Cohere, Google AI)
- ✅ Native SDKs (Python, Node.js, .NET)
- ⚠️ Less specialized for code

**Pricing:**

- **Base:** $49/month
- **Includes:** Base data storage/request limits
- **Overage:** Charges for additional usage (contact for details)

**C++ Copilot Suitability: ⭐⭐⭐**

---

### 7. Credal

**URL:** https://credal.ai

**Technology:**

- ✅ Enterprise security
- ✅ Slack integration
- ✅ Comprehensive APIs
- ⚠️ Less code-specific features

**Pricing:**

- **Base:** $500/month
- **Model:** Enterprise pricing
- **Includes:** Base data/API limits, enterprise support

**C++ Copilot Suitability: ⭐⭐⭐**

---

### 8. CustomGPT

**URL:** https://customgpt.ai

**Technology:**

- ✅ Custom GPT models
- ⚠️ Business/customer engagement focus
- ⚠️ Limited code-specific features

**Pricing:**

- **Base:** $89/month
- **Includes:** Base data storage/request limits
- **Overage:** May apply (contact for details)

**C++ Copilot Suitability: ⭐⭐**

---

### 9. Circlemind ⭐ **GRAPH RAG SPECIALIST**

**URL:** https://circlemind.co/rag

**Technology:**

- ✅ **GraphRAG system** (knowledge graphs + vector databases)
- ✅ Promptable GraphRAG (adapts to specific use cases)
- ✅ Entity relationship understanding
- ✅ Multi-hop reasoning via knowledge graphs
- ✅ Code relationship mapping (functions, classes, dependencies)

**Pricing:**

- **Community Edition:** Free (self-hosted, open-source)
- **Business:** $300/month (fully managed, advanced features, premium support)
- **Enterprise:** Custom pricing (large-scale deployments)

**C++ Copilot Suitability: ⭐⭐⭐⭐⭐**

- Excellent for understanding code relationships, dependencies, and entity connections
- Knowledge graphs ideal for code structure (class hierarchies, function calls, includes)

---

### 10. Lettria ⭐ **GRAPH RAG FOR CODE**

**URL:** https://lettria.com (estimated)

**Technology:**

- ✅ **GraphRAG solution** (mentioned in original sources)
- ✅ Precision and dependability for generative AI
- ✅ Graph-based retrieval for complex code relationships
- ✅ Entity relationship understanding

**Pricing:**

- **€600/month** (~$650/month)

**C++ Copilot Suitability: ⭐⭐⭐⭐**

- GraphRAG optimized for code understanding and relationships

---

## Recommendations

### For Graph RAG Support:

**Best Graph RAG: Circlemind** - Dedicated GraphRAG service at $300/month, excellent for code relationships and entity understanding

**Graph RAG + Code: LlamaCloud** - Graph RAG via NebulaGraph integration, plus native code support and GitHub integration

**Graph RAG Budget: Lettria** - GraphRAG solution at €600/month, optimized for code precision

### General Recommendations:

**Primary: LlamaCloud** - Best for native code support, Graph RAG, AST parsing, GitHub integration, sophisticated prompts

**Graph RAG Specialist: Circlemind** - Best for knowledge graph-based retrieval, entity relationships, multi-hop reasoning

**Secondary: Progress Agentic RAG** - Best for enterprise security, modular architecture, any LLM support

**Budget: Ragie** - Best for hybrid search at $100-500/month (no Graph RAG)

**Quick Start: Vectara** - Best for free tier testing and high accuracy (no Graph RAG)

---

## Implementation Considerations

- **Data Ingestion:** Code parsing (C++/C with syntax awareness), documentation formats, GitHub/Slack integration
- **Graph RAG Setup:** Configure knowledge graph extraction for code entities (classes, functions, dependencies)
- **Hybrid Retrieval:** Graph traversal + Vector search (semantic) + Keyword search (exact matches) + Re-ranking
- **System Prompts:** Code context, C++ specificity, multi-source combination, QA/summarization optimization

---

## Sources

1. https://slashdot.org/software/rag-as-a-service-ragaas/
2. https://www.progress.com/agentic-rag
3. https://llamaindex.ai/llamacloud
4. https://vectara.com
5. Web search results for RAGaaS market analysis

---

_Report Generated: December 2025_
