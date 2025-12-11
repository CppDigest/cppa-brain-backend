# RAGaaS Market Report for C++ Copilot

## Executive Summary

This report analyzes RAGaaS platforms for a **C++ Copilot** use case requiring support for code files (C++, C), documentation (HTML, JSON, PDF, MD, TXT, ADOC), hybrid retrieval methods, sophisticated system prompts, and high accuracy for QA and summarization.

---

## C++ Copilot Requirements

- **Data Types:** Code files (cpp, hpp, c, h), HTML, JSON, PDF, Markdown, TXT, AsciiDoc
- **Data Sources:** Documentation, GitHub, HuggingFace, compiler outputs, Slack, mailing lists, Reddit
- **Capabilities:** QA, Summarization, Hybrid retrieval (vector + keyword + semantic), High accuracy, Sophisticated system prompts

---

## Recommended Platforms

### 1. LlamaCloud (by LlamaIndex) ⭐ **BEST FIT**

**URL:** https://llamaindex.ai/llamacloud

**Technology:**

- ✅ Native code parsing with AST support
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

## Comparison Summary

| Platform             | Code Support | Hybrid Retrieval | System Prompts | GitHub/Slack | Accuracy   | Pricing           | Recommendation |
| -------------------- | ------------ | ---------------- | -------------- | ------------ | ---------- | ----------------- | -------------- |
| **LlamaCloud**       | ⭐⭐⭐⭐⭐   | ✅ Yes           | ✅ Advanced    | ✅ Native    | ⭐⭐⭐⭐⭐ | Usage-based       | **BEST FIT**   |
| **Progress Agentic** | ⭐⭐⭐⭐     | ✅ Yes           | ✅ Advanced    | ⚠️ Custom    | ⭐⭐⭐⭐   | $700-1,750/mo     | **STRONG**     |
| **Ragie**            | ⭐⭐⭐⭐     | ✅ Yes           | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐⭐   | $100-500/mo       | **GOOD**       |
| **Vectara**          | ⭐⭐⭐       | ✅ Yes           | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐⭐⭐ | Free tier + usage | **SOLID**      |
| **Nuclia**           | ⭐⭐⭐       | ✅ Yes           | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐⭐   | Contact           | **MULTIMODAL** |
| **Graphlit**         | ⭐⭐⭐       | ⚠️ Limited       | ✅ Custom      | ⚠️ Custom    | ⭐⭐⭐     | $49/mo            | **BUDGET**     |
| **Credal**           | ⭐⭐         | ⚠️ Limited       | ⚠️ Limited     | ✅ Slack     | ⭐⭐⭐     | $500/mo           | **SECURITY**   |
| **CustomGPT**        | ⭐⭐         | ⚠️ Limited       | ⚠️ Limited     | ❌ No        | ⭐⭐       | $89/mo            | **NOT IDEAL**  |

---

## Pricing Factors

**How pricing relates to usage:**

1. **Data Size:**

   - Fixed plans (Progress): Includes base data (10GB), upgrade for more
   - Usage-based (LlamaCloud, Vectara): Charge per GB indexed
   - Tier-based (Ragie): Different limits per tier

2. **Request Count:**

   - Token-based (Progress): $0.008 per token after allowance
   - Query-based (Vectara, LlamaCloud): Per query/request
   - Tier-based: Monthly limits with overage charges

3. **Service Type:**
   - Higher tiers: More features/support (e.g., Progress Growth $1,750 vs Fly $700)
   - Enterprise: Custom pricing for large deployments
   - Free tiers: Vectara, LlamaCloud for testing

**Cost Optimization:**

- High token usage → Query-based pricing preferred
- Large data volumes → Enterprise pricing or usage-based storage
- Variable usage → Usage-based pricing
- Predictable usage → Fixed monthly plans

---

## Recommendations

**Primary: LlamaCloud** - Best for native code support, AST parsing, GitHub integration, sophisticated prompts

**Secondary: Progress Agentic RAG** - Best for enterprise security, modular architecture, any LLM support

**Budget: Ragie** - Best for hybrid search at $100-500/month

**Quick Start: Vectara** - Best for free tier testing and high accuracy

---

## Implementation Considerations

- **Data Ingestion:** Code parsing (C++/C with syntax awareness), documentation formats, GitHub/Slack integration
- **Hybrid Retrieval:** Vector search (semantic) + Keyword search (exact matches) + Re-ranking
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
