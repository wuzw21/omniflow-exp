# PAPT Page-Embedding Benchmark

This experiment is evaluation-only: do not train or fine-tune OmniTransfer for
it. The canonical builder, evaluator, full protocol, archive hashes, and
2026-08-26 result live in the canonical OmniTransfer checkout:

- `scripts/analysis/build_papt4_retrieval_test.py`
- `scripts/analysis/evaluate_papt_page_embeddings.py`
- `docs/papt_frozen_page_retrieval_20260826.md`

Canonical checkout: `~/Projects/Omni/OmniTransfer`.

The frozen protocol uses 716 valid PAPT4 pairs from 94 apps with zero overlap
against the 433 apps in the current V10 PAPT provenance. Each `screen_1` query
ranks all 716 `screen_2` targets. The manifest SHA-256 is
`61d75dd4992196a0f8329519cb13f0ee7ba62940ff510867bc0b48c1d3a660d4`.

The only compared methods are:

1. Screenshot Embedding: Qwen3-VL-Embedding, 1024D, cloud API.
2. XML Embedding: MobileGPT compression plus GLM-Embedding-2, cloud API.
3. OmniTransfer V10: frozen checkpoint, 1024D page readout, local CPU, K=16.

The completed strict exact-pair Top-1 results are 79.61%, 25.42%, and 29.47%,
respectively. Full Top-3/Top-5, cosine-score margins, wall time, request usage,
and cost are recorded in the canonical OmniTransfer document. Raw PAPT data,
embedding caches, credentials, and result artifacts stay outside this repo.
