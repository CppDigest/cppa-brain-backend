# LangChain RAG Resource Usage - Quick Reference

## 🎯 At a Glance

| Metric | Value | Notes |
|--------|-------|-------|
| **Typical RAM** | 1.5-2GB | 50K documents |
| **Query CPU** | 15-30% | Single query |
| **Query Time (uncached)** | 250ms | Average |
| **Query Time (cached)** | 5ms | 65-80% hit rate |
| **Index Time** | 15-20 min | 50K documents |

---

## 💾 Memory Components

```
Total RAM: ~1.8GB
├─ ChromaDB:          800MB  (44%)
├─ Embedding Model:   300MB  (17%)
├─ Query Cache:       150MB  (8%)
├─ TF-IDF:           100MB  (6%)
├─ BM25 (dynamic):    150MB  (8%)
└─ Application:       300MB  (17%)
```

---

## ⚡ Performance Profiles

### Low Memory Mode (< 2GB RAM)
```yaml
dense_top_k: 30
sparse_top_k: 30
max_bm25_docs: 10
cache_max_memory_entries: 500
```
**Result**: ~1.2-1.5GB RAM, +50ms query time

### Balanced Mode (2-4GB RAM) ⭐ Recommended
```yaml
dense_top_k: 50
sparse_top_k: 50
max_bm25_docs: 15
cache_max_memory_entries: 1000
```
**Result**: ~1.8-2.2GB RAM, 250ms query time

### High Performance Mode (4GB+ RAM)
```yaml
dense_top_k: 100
sparse_top_k: 100
max_bm25_docs: 50
cache_max_memory_entries: 2000
```
**Result**: ~2.5-3GB RAM, 180ms query time

---

## 📊 Scaling Guidelines

| Documents | Min RAM | Recommended RAM | Query Time | Index Time |
|-----------|---------|-----------------|------------|------------|
| 10K | 1GB | 2GB | 180ms | 3 min |
| 50K | 1.5GB | 4GB | 250ms | 15 min |
| 100K | 2GB | 8GB | 350ms | 35 min |
| 500K+ | 4GB | 16GB+ | 600ms | 3+ hours |

---

## 🐳 Docker Resource Limits

### Small Deployment
```yaml
resources:
  limits:
    memory: "2Gi"
    cpu: "1"
  reservations:
    memory: "1Gi"
    cpu: "0.5"
```

### Medium Deployment
```yaml
resources:
  limits:
    memory: "4Gi"
    cpu: "2"
  reservations:
    memory: "2Gi"
    cpu: "1"
```

### Large Deployment
```yaml
resources:
  limits:
    memory: "8Gi"
    cpu: "4"
  reservations:
    memory: "4Gi"
    cpu: "2"
```

**Last Updated**: October 27, 2024  
**Quick Access**: Keep this handy for production deployments!

