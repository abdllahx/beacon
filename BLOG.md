# Beacon — verifying news claims with satellite imagery and a 21-node agent DAG

*A multimodal RAG system that takes a news claim, pulls before/after Sentinel imagery, and tells you whether the event actually happened — with cited evidence, not vibes.*

---

## The problem

In 2026, four jobs share a question and none of them have a good answer:

- A newsroom editor receives a wire claim that a wildfire destroyed a village.
- An OSINT analyst sees a Telegram post showing alleged battlefield damage.
- An insurance adjuster gets a flood claim against a policy in rural India.
- A hedge fund analyst hears a rumor about a stalled construction project.

All four want the same thing: **did this actually happen, where, and at what scale?** And all four currently get the answer either from slow manual analysis (Bellingcat-style geolocation takes hours) or from a text-only LLM that confidently hallucinates coordinates.

Beacon closes the loop end-to-end: claim → geolocation → satellite imagery diff → grounded report.

## What it actually does

```
news article → NER + zero-shot relevance filter → Nominatim + LLM-disambiguated geocoder
            → Sentinel-2 + Sentinel-1 SAR before/after
            → NBR/dNBR + SAR backscatter change + SegFormer + DETR + SigLIP
            → Claude vision verdict + BART-CNN summary + multilingual MarianMT report
```

All composed as a **21-node LangGraph DAG with SQLite checkpointing** — every run is resumable, every span is observable.

## The 9-task multimodal stack

The brief required nine distinct HuggingFace tasks "genuinely composed, not bolted on." Every task below fires on every run and persists results into Postgres. You can audit it from the database, not just the README:

| Task | Model | Where it runs |
|---|---|---|
| Token Classification (NER) | `Davlan/bert-base-multilingual-cased-ner-hrl` | extract_claim node |
| Zero-Shot Text Classification | `facebook/bart-large-mnli` | extract_claim node |
| Sentence Similarity | `all-MiniLM-L6-v2` | embed-claims (pgvector 384) |
| Summarization | `facebook/bart-large-cnn` | summarize_article node |
| Translation | `Helsinki-NLP/opus-mt-en-{es,ar,fr}` | translate_report node |
| Image Segmentation | `nvidia/segformer-b0-finetuned-ade-512-512` | segment_after node |
| Object Detection | `facebook/detr-resnet-50` | detect_after node |
| Zero-Shot Image Classification | `google/siglip-base-patch16-224` | classify_tile node |
| Visual Document Retrieval | SigLIP image+text embeddings | vdr_search (pgvector 768) |

Plus Claude Sonnet via the Agent SDK as the per-tile vision-language planner and the verification synthesizer.

## The hard parts

### 1. Time-indexed raster retrieval is not document RAG

Conventional RAG chunks a document corpus and runs cosine similarity. Beacon retrieves *time-indexed raster tiles* across two sensor modalities (Sentinel-2 optical, Sentinel-1 SAR) with cloud-cover constraints, intersection-over-target ≥ 0.7 coverage filtering, and ±30/±14 day before/after windows.

The Sentinel-2 STAC search returns scenes that *intersect* the bounding box. Naively, ~60% of returned scenes cover under 5% of the target — you get triangular slivers along granule edges that look like data but verify nothing. Coverage-aware item selection is mandatory, not optional.

### 2. The TCI lie

Beacon's first version used Sentinel-2's TCI (true-color imagery) asset for Claude vision analysis and produced an embarrassing failure: it called the 2024 Park Fire (California, 400k acres burned) "inconclusive" with confidence 0.40. Both burn scars and dense conifer canopy render dark in TCI. To a human eye looking at the after-tile, it's ambiguous. To a vision model, it's noise.

The fix was to compute the **Normalized Burn Ratio**: `NBR = (B08 - B12) / (B08 + B12)` — the SWIR band (B12) is the standard remote-sensing signal for burned ground. Then take the delta: `dNBR = NBR_before - NBR_after`. Pixels above the Key & Benson 0.27 moderate-burn threshold get classified as burned and counted as a percentage. Park Fire flipped from inconclusive (0.40) to supported (0.82) without any prompt changes — the right inputs were missing, not the right reasoning.

### 3. Geocoding ambiguity is where text-only LLMs die

"A wildfire ravaged Canoe yesterday." Where is Canoe? Nominatim returns Canoe (Kentucky) first because it's the most populated. But the article context says British Columbia. A naive geocoder + LLM stack hands the wrong bbox to the imagery layer and the verifier dutifully reports "no fire visible" — because there wasn't one in Kentucky.

Beacon's geocoder pulls top-3 NER LOC entities, queries Nominatim for top-3 candidates per entity, and asks Claude to disambiguate against the article context. Multi-location fallback (try entity 1, 2, 3 until one passes Claude's article-context check) is required for usable recall on real news data — without it the pipeline falls over on locations like "Hell Gate" (which Nominatim sends to Manhattan) or "Me" (which goes to Montenegro).

### 4. Honest evaluation against peer-reviewed ground truth

Most agent demos report vibes. Beacon reports **Acc@N km** following Gritta et al. 2018's geoparsing evaluation standard, against **GDIS** (Rosvold & Buhaug 2021, *Sci. Data*) — peer-reviewed centroids for 39,953 disaster locations.

Current numbers on N=42 GDIS-validated events:

- **Acc@10 km / 50 km / 161 km: 16.7% / 47.6% / 64.3%**
- **Median centroid distance: 60.7 km**
- **Recall (supported): 23.8%**, refuted 16.7%, inconclusive 59.5%

Note the inconclusive rate is a feature, not a bug — the verifier correctly flags low-evidence cases (e.g., earthquakes don't show in 10 m optical/SAR change) instead of fabricating verdicts. A snapshot/diff machinery (`beacon eval-snapshot|eval-diff`) lets every iteration's metrics be tracked over time.

### 5. The synthetic-text caveat

The current eval feeds the pipeline text synthesized from EM-DAT structured fields, not real news articles. Real news has 10× richer location context, multiple supporting LOC entities, and date precision the EM-DAT fields lack. The Acc@10km number under-reports the pipeline's true geoparsing capacity — a real-news GDELT-scraping benchmark is the next iteration.

## Production posture

Built on free infra deliberately, to prove the architecture rather than throw money at it:

- **Postgres 16 + PostGIS 3.5 + pgvector 0.8** for hybrid retrieval. `claims.embedding vector(384)` for sentence dedup, `tile_archive.embedding vector(768)` HNSW for VDR. Single store, no Pinecone-plus-PostGIS juggling.
- **Microsoft Planetary Computer** for Sentinel imagery, anonymous-readable STAC.
- **Claude Sonnet via Agent SDK** instead of self-hosted Qwen2.5-VL — no GPU bill, swappable later if budget changes.
- **Langfuse Cloud (Hobby tier)** for tracing — every Claude/HF call gets a span, the full agent DAG renders as a timeline.
- **DIY cost log** — every Claude call writes to `cost_events` with input/output token estimates and per-operation aggregation. Median per-run cost is queryable from the dashboard.
- **HITL feedback table** — analyst confirms or corrects each verdict in the Streamlit UI. Corrections route into a DSPy training set for next-iteration prompt optimization.
- **DSPy synthesis prompt layer** — `dspy.Signature` over the verifier prompt with cached high-quality past runs as few-shot demos. The optimizer is wired but training waits for HITL labels to accumulate.

## What I'd build next

1. **Real-news benchmark via GDELT scraping** — kills the synthetic-text caveat. Should bump Acc@10km ~2-3×.
2. **Scale eval to N=500 GDIS events** — the spec target. Architecture is ready; only compute budget gates it.
3. **Active-learning loop** — runs with low-confidence verdicts get prioritized for HITL review, and corrected verdicts retrain the DSPy synthesizer weekly.
4. **Conflict-event coverage** — ACLED loader is built but their Open tier doesn't expose event-level API. Pending researcher-tier upgrade.

## Why this beats the AI-wrapper glut

Beacon is multimodal because the problem requires it — text alone cannot tell you whether a wildfire happened. The retrieval problem (time-indexed raster + text + geo) is genuinely hard. The agent planning is real (the system decides which imagery to pull, when, and what to compare). And the eval has objective ground truth: you know if the wildfire happened.

In 2026 the AI-engineering job market is full of "I built a RAG chatbot" portfolios. The differentiator is depth in one project that hits all the production bullets — RAG at scale, agent orchestration, evaluation frameworks, multimodal, observability, cost controls, HITL — with real artifacts behind each. That's Beacon.

---

**Code:** [github.com/abdllahx/beacon](https://github.com/abdllahx/beacon)
**Demo:** [beacon-rt7qkydjptf7vm8ccpua5r.streamlit.app](https://beacon-rt7qkydjptf7vm8ccpua5r.streamlit.app/)
**License:** MIT.
