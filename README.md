# Beacon - Multimodal Geospatial Event Verification

[![tests](https://github.com/abdllahx/beacon/actions/workflows/test.yml/badge.svg)](https://github.com/abdllahx/beacon/actions/workflows/test.yml)
[![demo](https://img.shields.io/badge/live%20demo-streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://beacon-rt7qkydjptf7vm8ccpua5r.streamlit.app/)

**Live demo:** https://beacon-rt7qkydjptf7vm8ccpua5r.streamlit.app/

A production-grade agent system that detects claims in global news (natural disasters, conflict events, deforestation, humanitarian crises) and autonomously verifies or refutes them by cross-referencing recent satellite imagery, producing cited, evidence-backed verification reports.

**Closes the loop:** *claim → geolocation → imagery diff → grounded report.*

Today this work is a slow mix of manual OSINT (Bellingcat-style) and unreliable text-only LLMs that hallucinate coordinates. Beacon does it in minutes, with a multimodal evidence stack you can audit.

---

## What it does

```
news article → NER + relevance filter → geocoder (LLM-disambiguated)
              → Sentinel-2 + Sentinel-1 SAR before/after
              → NBR/dNBR + SAR-change + SegFormer + DETR + SigLIP
              → Claude vision verdict + BART summary + multilingual report
```

All composed as a **21-node LangGraph DAG with SQLite checkpointing** so any run is resumable.

## 9 HuggingFace tasks composed end-to-end

The original Beacon spec called for 9 distinct HF tasks "genuinely composed, not bolted on." Every task below fires on every run and persists results into the database — provable from the verification dashboard, not just claimed in a README.

| HF Task | Model | Where it runs | Persists into |
|---|---|---|---|
| Token Classification (NER) | `Davlan/bert-base-multilingual-cased-ner-hrl` | `extract_claim` node | `claims.locations` |
| Zero-Shot Text Classification | `facebook/bart-large-mnli` | `extract_claim` node | `claims.event_type` |
| Sentence Similarity | `sentence-transformers/all-MiniLM-L6-v2` | `embed-claims` CLI | `claims.embedding` (pgvector 384) |
| Summarization | `facebook/bart-large-cnn` | `summarize_article` node | `verification_runs.article_summary` |
| Translation | `Helsinki-NLP/opus-mt-en-{es,ar,fr}` | `translate_report` node | `verification_runs.translations` |
| Image Segmentation | `nvidia/segformer-b0-finetuned-ade-512-512` | `segment_after` node | `imagery_metadata.s2.after.segmentation` |
| Object Detection | `facebook/detr-resnet-50` | `detect_after` node | `imagery_metadata.s2.after.detections` |
| Zero-Shot Image Classification | `google/siglip-base-patch16-224` | `classify_tile` node | `imagery_metadata.s2.after.zero_shot` |
| Visual Document Retrieval | `google/siglip-base-patch16-224` (image+text embeddings) | `vdr_search` node | `imagery_metadata.vdr_matches` (pgvector 768) |

Plus **Claude Sonnet via the Agent SDK** for the high-level vision VQA + verification synthesizer.

## Architecture

```mermaid
graph TD;
    A[News article] --> B[extract_claim<br/>NER + zero-shot]
    B --> C[geocode_claim<br/>Nominatim + LLM disambiguation]
    C --> D[load_claim] --> E[init_run]
    E --> F[fetch_s2_before]
    E --> G[fetch_s2_after]
    E --> H[fetch_nbr_before]
    E --> I[fetch_nbr_after]
    E --> J[fetch_s1_before]
    E --> K[fetch_s1_after]
    H --> L[compute_dnbr]
    I --> L
    J --> M[compute_s1_change]
    K --> M
    G --> N[segment_after]
    G --> O[detect_after]
    G --> P[classify_tile]
    G --> Q[vdr_search]
    F --> R[vision_vqa]
    G --> R
    L --> R
    M --> R
    N --> R
    O --> R
    P --> R
    Q --> R
    R --> S[persist_vision]
    S --> T[synthesize]
    S --> U[summarize_article]
    T --> V[translate_report]
    U --> V
    V --> W[END]
```

The full DAG (`uv run beacon graph-render`) has **21 nodes** with **6-way parallel imagery fetch** + **4-way parallel vision-task fan-out**.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph + SqliteSaver checkpointer | Branching retries, resumable runs |
| Storage | Postgres 16 + PostGIS 3.5 + pgvector 0.8 | Hybrid geo + dense retrieval, single store |
| Imagery | Microsoft Planetary Computer (Sentinel-2 L2A + Sentinel-1 RTC) | Free, anonymous-readable STAC |
| Ground truth | NASA FIRMS + EM-DAT (CRED) + GDIS (Rosvold & Buhaug 2021) + ACLED (loader shipped, awaits API tier) | Triangulated; GDIS is peer-reviewed; ACLED OAuth flow validated end-to-end pending researcher tier approval |
| LLM planner/synthesizer | Claude Sonnet via Agent SDK | Available on Max plan, no API budget |
| Vision/text models | 9 HuggingFace tasks (table above) | All via free Inference Providers or local CPU |
| Demo UI | Streamlit | Single-file, all 9 modalities visible |

## Eval methodology

The benchmark feeds the pipeline **only synthesized article text** (built from EM-DAT structured fields). Extract → geocode → imagery → vision all run from text alone, and the resulting bbox is compared against **GDIS peer-reviewed centroids** (Rosvold & Buhaug 2021, *Sci. Data*).

We report Accuracy@N km following the geoparsing literature standard (Gritta et al. 2018, *A Pragmatic Guide to Geoparsing Evaluation*) — the percentage of predicted locations within N km of ground truth, at N ∈ {10, 50, 161} km.

See [EVAL.md](EVAL.md) for the full methodology, prior-version critique, and snapshot/diff machinery.

## Demo

5 hand-curated demo events ship with the project:

| Event | Verdict | dNBR burn % | SAR backscatter Δ | SigLIP top class |
|---|---|---:|---:|---|
| 2025 Pacific Palisades Fire | supported (0.85) | 37.7% | -15.0% | "burned land or fire scar" |
| 2024 Park Fire (CA) | supported (0.82) | 19.0% | — | (varies) |
| 2023 Lahaina (Maui) | supported (0.78) | 1.7% | — | (varies) |
| 2023 Donnie Creek (BC) | inconclusive (0.40) | 0% | — | (varies) |
| 2023 Rhodes (Greece) | inconclusive (0.40) | 0% | — | (varies) |

**Live:** [beacon-rt7qkydjptf7vm8ccpua5r.streamlit.app](https://beacon-rt7qkydjptf7vm8ccpua5r.streamlit.app/) — backed by Neon Postgres + Langfuse Cloud, HITL writes password-locked.

**Local:** `uv run beacon ui` → http://localhost:8501

## Quickstart

```bash
# Prereqs: Docker, uv (https://docs.astral.sh/uv/), a free HuggingFace token, NASA FIRMS key, NewsAPI key.
git clone https://github.com/abdllahx/beacon.git
cd beacon

# Postgres (PostGIS + pgvector — built from db/Dockerfile, arm64-friendly)
docker compose up -d
docker exec -i beacon-db psql -U beacon -d beacon < sql/001_init.sql
# (apply 002–013 likewise)

# Python
uv sync

# Configure .env  (example provided as .env.template)
cp .env.template .env
# edit .env to add HF_TOKEN, NASA_FIRMS_KEY, NEWSAPI_KEY

# Seed demo events + ground truth
uv run beacon demo-seed
uv run beacon firms-load --area "-141,48,-115,60" --days 5
uv run beacon emdat-load
uv run beacon gdis-load

# Run the full pipeline on a demo event
uv run beacon graph-run 11  # claim_id 11 = Palisades

# Open the dashboard
uv run beacon ui
```

## Production posture

| | |
|---|---|
| Tracing | Langfuse Cloud — every Claude + HF call traced via `@observe` |
| Cost | DIY log → `cost_events` table, per-op USD aggregation in dashboard |
| Latency | `beacon latency-report` → **p50 219s · p95 585s** over n=50 runs |
| HITL | password-gated thumbs/correction flow → `feedback` table → DSPy demos |
| Prompt opt | `dspy.Signature` over verifier prompt, few-shot bootstrap from labeled runs |
| Hosting | Streamlit Cloud + Neon Postgres + Langfuse Cloud — all free tier |

See [DEPLOY.md](DEPLOY.md) for the deploy path. See [BLOG.md](BLOG.md) for the writeup of hard parts (time-indexed retrieval, the TCI lie, geocoding ambiguity, honest evaluation).

## License

MIT.
