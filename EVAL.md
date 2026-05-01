# Beacon evaluation methodology

This document describes how Beacon is benchmarked, what numbers we publish, and the
honest tradeoffs behind them.

## Ground truth

| Source | Coverage | License |
|---|---|---|
| **GDIS** (Rosvold & Buhaug 2021, *Sci. Data*) | 9,924 disasters 1960–2018 with peer-reviewed subnational centroids | CC BY 4.0 |
| **EM-DAT** (CRED / UCLouvain) | 27,000+ disasters 1900–present with country/admin metadata | Free for research |
| **NASA FIRMS** | Real-time fire detections (last 60 days) | Public |

We use a SQL view `disaster_ground_truth` that prefers GDIS centroids and falls back
to EM-DAT's own coordinates. Reports are stratified by `gt_source`, so the
GDIS-validated subset is always cleanly separable as the rigorous-methodology
headline number.

## Benchmark methodology — v2 (current)

For each EM-DAT event in the sample, we synthesize a short news-style article from
the structured fields:

```
Wildfire devastates Pernik. A wildfire struck Pernik in Bulgaria on July 30, 2018,
killing at least 5 people and affecting an estimated 1,200 residents.
```

The pipeline receives **only** that text. There is no shortcut — `extract_claim`,
`geocode_claim`, imagery fetch, and synthesis all run from text alone. The
**reconstructed bbox** is what gets scored against GDIS ground truth.

### What v1 did wrong

The first benchmark version copied EM-DAT's own bbox into the synthetic claim, then
"compared" predicted bbox to ground-truth bbox. IoU was trivially 1.0 and centroid
distance was trivially 0. Numbers looked great but meant nothing. v2 fixes this; the
fix is in [`benchmark.py`](src/beacon/benchmark.py) (search for
`_synthesize_article` — note it inserts only an article row, no claim).

## Metrics

### Geoparsing (the headline)

Following Gritta et al. 2018, *A Pragmatic Guide to Geoparsing Evaluation* (LREV):

- **Accuracy@N km** — % of predicted centroids within N km of ground truth.
  Reported at N ∈ {10, 50, 161}. 161 km is the literature convention
  (~Earth radius / 40).
- **Mean / median centroid distance (km)** — haversine between predicted bbox
  centroid and GDIS centroid.

### Spatial (secondary)

- **Mean bbox IoU** — `ST_Intersection / ST_Union` of the predicted bbox vs the
  GDIS-derived bbox. Note GDIS bboxes are admin-region scale and often much larger
  than the predicted bbox, so IoU under-rewards even good predictions; we report it
  for completeness but Accuracy@N is the cleaner signal.

### Verdict

- **Recall on supported** — % of true-positive events the pipeline labels supported.
- **Inconclusive rate** — % the pipeline honestly declines to verify (the desired
  behavior on events the modality can't capture, e.g., earthquakes via 10 m optical).

## Snapshot/diff machinery

Every benchmark run produces a JSON snapshot under
`data/eval_snapshots/{timestamp}_{label}.json` containing:
- timestamp + git SHA + dirty flag
- the exact tunable values active for the run
- the full eval-report metrics
- per-event verdict + centroid distance + IoU

Snapshots are diffable via `uv run beacon eval-diff a b`, which surfaces:
- metric-by-metric deltas (Accuracy@N km, mean distance, IoU, etc.)
- per-event verdict changes (events whose verdict moved between snapshots)
- distance regressions (events where centroid distance moved by >5 km)

This turns *"are we moving in the right direction?"* from a vibe-check into a
tracked, version-controlled artifact.

## Why these metrics, not F1

The original Beacon spec asked for "verification F1." A clean F1 requires
**negative examples** — events that *did not happen* — and we don't have them in
EM-DAT/GDIS by construction (those datasets are positive-only). Two paths to F1:

1. **Date-shifted negatives** — generate "Wildfire in X 5 years before the real
   event" claims and expect the pipeline to refute. Doable, planned for Month 3.
2. **Real news scraping** — pull articles from GDELT/NewsAPI by date+country+type.
   Some will be misleading or misattributed; the pipeline should refute those.

Until then, Accuracy@N km + verdict-recall is what we publish, and the synthetic
caveat is on the README.

## Caveats we acknowledge openly

1. **Synthetic article text vs real news.** EM-DAT-derived stub text is far less
   information-dense than a 600-word real news article. NER and the geocoder will
   under-perform on stubs vs. real news. Real numbers should *improve* on real
   inputs; current numbers are a worst-case lower bound.

2. **GDIS doesn't cover wildfires.** Floods, storms, earthquakes, landslides,
   droughts, volcanic, extreme temperature — yes. Wildfires — no. The
   GDIS-validated benchmark is therefore non-wildfire by construction. Wildfire
   recall is reported separately on the 5 hand-curated demo events (Palisades, Park
   Fire, Maui, Donnie Creek, Rhodes) using FIRMS as supplementary ground truth.

3. **GDIS bboxes are admin-region scale.** A "flood in Kigali" GDIS bbox is the
   entire city. The predicted bbox from the pipeline can be smaller and more
   precise; IoU will under-credit this. Centroid distance is the more honest signal.

4. **DETR at Sentinel-2 resolution.** COCO classes are sub-pixel at 10 m. The
   `detect_after` node returns 0 objects on most tiles. This is an honest
   resolution limit, not a pipeline bug — the node and its overlay path are wired
   so a sub-meter input drops in transparently.

5. **The 5 demo events are hand-curated.** They are chosen for visual impact, not
   benchmark sampling. Demo numbers ≠ benchmark numbers, and we keep them
   separate.

## Reproducing the benchmark

```bash
uv run beacon eval-build --n 50 --types "Flood,Storm,Earthquake" --gdis-only
uv run beacon eval-run                 # ~2 hours; per-event progress lines
uv run beacon eval-snapshot v4-N50     # captures the JSON snapshot
uv run beacon eval-report --gdis-only  # human-readable summary
uv run beacon eval-diff v3-prev v4-N50 # diff vs prior snapshot
```
