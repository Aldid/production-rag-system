# Production RAG System — Benchmark Evaluation Report

**Total Test Cases:** 100  
**Date:** 2026-10-09 14:03:51 UTC  
**Architecture:** Hybrid Dense-Sparse Retrieval (deterministic hashed-token embeddings + BM25) with Reciprocal Rank Fusion (RRF)  
**Corpus:** 5 short self-written reference documents; the 100 queries and expected answers were also written by the author (see `src/evals/dataset.py`). Hit-Rate is measured at document level, so a random ranking already scores ~20% Hit-Rate @ 1.  
**Faithfulness** is a lexical proxy (share of answer tokens found in the retrieved context), not an LLM- or human-judged score.  
**Answer relevance** (token Jaccard vs. ground-truth answer): 13.64%

---

## 1. Executive Metrics Summary

| Metric | Measured Value | Production Target | Status |
| :--- | :---: | :---: | :---: |
| **Hit-Rate @ 1** | **96.00%** | $\ge 80.0\%$ | **PASS** |
| **Hit-Rate @ 3** | **99.00%** | $\ge 95.0\%$ | **PASS** |
| **Hit-Rate @ 5** | **100.00%** | $\ge 98.0\%$ | **PASS** |
| **Mean Reciprocal Rank (MRR)** | **0.9770** | $\ge 0.8500$ | **PASS** |
| **Avg Keyword Grounding** | **92.50%** | $\ge 85.0\%$ | **PASS** |
| **Context Faithfulness** | **97.29%** | $\ge 90.0\%$ | **PASS** |
| **Citation Precision** | **99.00%** | $\ge 95.0\%$ | **PASS** |
| **Average Query Latency** | **1.31 ms** | $\le 25.0$ ms | **PASS** |
| **P95 Query Latency** | **1.73 ms** | $\le 50.0$ ms | **PASS** |

---

## 2. Category Performance Breakdown

| Category | Cases | Hit-Rate @ 1 | Hit-Rate @ 3 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **API Gateway** | 20 | 100.0% | 100.0% | 1.0000 |
| **Databases** | 20 | 100.0% | 100.0% | 1.0000 |
| **Security** | 20 | 95.0% | 100.0% | 0.9750 |
| **Infrastructure** | 20 | 95.0% | 100.0% | 0.9750 |
| **Browser Automation** | 20 | 90.0% | 95.0% | 0.9350 |

---

## 3. Failure Cases & Outliers
Total identified edge/failure cases: **6**

- **eval_067** (Infrastructure): *"What is the maximum latency for cache key purging after an update?"* — Reason: `Keyword coverage low (0/2)`
- **eval_076** (Infrastructure): *"Do cache keys have indefinite lifetimes?"* — Reason: `Keyword coverage low (0/1)`
- **eval_082** (Browser Automation): *"Why does Playwright persist browser contexts to disk?"* — Reason: `Keyword coverage low (0/2)`
- **eval_094** (Browser Automation): *"How are cookies stored securely for persistent browser contexts?"* — Reason: `Keyword coverage low (0/1)`
- **eval_095** (Browser Automation): *"What event detects outgoing network requests in CDP?"* — Reason: `Keyword coverage low (0/1)`
- **eval_096** (Browser Automation): *"What event captures incoming HTTP responses in CDP?"* — Reason: `Target doc 'doc_playwright_cdp' not in top 3 retrieved: ['doc_api_gateway', 'doc_security_guardrails', 'doc_distributed_cache', 'doc_api_gateway', 'doc_playwright_cdp']`
