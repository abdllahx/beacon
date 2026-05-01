"""Tunable algorithmic thresholds for the Beacon pipeline.

These values are NOT secrets and NOT typically environment-specific — they're
algorithm-level decisions that need a single source of truth. Every value below
has a rationale comment so reviewers can see *why* we picked it, traceable to
the literature where applicable.

To override at runtime for parameter sweeps:
    from beacon import tunables
    tunables.RELEVANCE_THRESHOLD = 0.4
    # ...then call into the pipeline.
"""

# ---------------------------------------------------------------------------
# Claim extraction (extract.py)
# ---------------------------------------------------------------------------

# Zero-shot classifier confidence required to keep an article. With 8 candidate
# labels the per-label softmax mass averages ~0.125, so 0.3 means "the top label
# clearly dominates" without being rigid. Lower = more recall but more noise.
RELEVANCE_THRESHOLD = 0.3

# Truncate article text to this many chars before NER + zero-shot. Long enough
# for a headline + lede; short enough to fit in BERT's 512-token context.
MAX_INPUT_CHARS = 2000


# ---------------------------------------------------------------------------
# Geocoding (geocode.py)
# ---------------------------------------------------------------------------

# Nominatim usage policy: max 1 req/s sustained. 1.1s adds a safety margin so
# we never get rate-limited on bursty calls.
# https://operations.osmfoundation.org/policies/nominatim/
NOMINATIM_REQUEST_INTERVAL_S = 1.1

# Minimum NER LOC-entity score to consider a candidate for geocoding.
# dslim/bert-base-NER scores real places > 0.9; 0.7 keeps everything except
# very ambiguous extractions like "Me" or single-letter token artifacts.
MIN_NER_SCORE = 0.7

# Multi-location single-Claude-call disambiguation: how many NER candidates do
# we fan out to Nominatim, and how many OSM matches per location do we gather.
# Total options sent to Claude = MAX × PER_LOC. Keep ≤ ~12 to fit cleanly in
# Claude's prompt and remain fast to disambiguate.
MAX_NER_CANDIDATES = 4
NOMINATIM_RESULTS_PER_LOC = 3


# ---------------------------------------------------------------------------
# Imagery — Sentinel-2 / NBR (imagery.py)
# ---------------------------------------------------------------------------

# Sentinel-2 native resolution is 10m. Below ~1km bbox the windowed read is
# pixel-thin and Claude can't see useful structure; above ~100km we cross MGRS
# tile boundaries and one item rarely covers the full target.
MIN_BBOX_DEG = 0.02
MAX_BBOX_DEG = 1.0

# Default fetch windows around the event date. 30 days back is enough to find
# a clear-sky pre-event Sentinel-2 pass (5-day revisit means ~6 candidates);
# 14 days forward catches most post-event passes before vegetation regrows.
DEFAULT_BEFORE_DAYS = 30
DEFAULT_AFTER_DAYS = 14

# Output PNG size (long edge). 1024 is large enough for Claude to discern
# burn-scar boundaries at 10m resolution while keeping file I/O fast.
DEFAULT_TILE_SIZE_PX = 1024


# ---------------------------------------------------------------------------
# dNBR — Key & Benson 2006 burn severity bands
# ---------------------------------------------------------------------------
# Reference: Key, C.H., Benson, N.C. (2006). Landscape Assessment: Ground measure
# of severity, the Composite Burn Index. USDA Forest Service General Technical
# Report RMRS-GTR-164-CD: LA 1-51.
#
# Severity bands:
#   < 0.10 : unburned
#   0.10 – 0.27 : low severity
#   0.27 – 0.44 : moderate-low
#   0.44 – 0.66 : moderate-high
#   ≥ 0.66 : high severity
DNBR_BAND_UNBURNED_MAX = 0.10
DNBR_BAND_LOW_MAX = 0.27
DNBR_BAND_MODERATE_LOW_MAX = 0.44
DNBR_BAND_MODERATE_HIGH_MAX = 0.66

# Headline burn-coverage % currently reports % pixels ≥ moderate-low threshold.
# Cleanup-4 (future) replaces this with full per-band counts.
DNBR_BURN_PCT_THRESHOLD = DNBR_BAND_LOW_MAX


# ---------------------------------------------------------------------------
# Sentinel-1 SAR (imagery.py)
# ---------------------------------------------------------------------------

# A 3 dB backscatter drop is the conventional threshold for "significant
# scattering change" in flood/burn detection — corresponds to ~50% drop in
# linear amplitude. Used to compute s1_decrease_pct.
S1_DECREASE_THRESHOLD_DB = 3.0


# ---------------------------------------------------------------------------
# Evaluation (eval_metrics.py)
# ---------------------------------------------------------------------------

# Geoparsing accuracy thresholds, following Gritta et al. 2018 "A pragmatic
# guide to geoparsing evaluation" (Lang. Resources Eval.). 161 km is the
# convention from the toponym-resolution literature (~ Earth radius / 40).
ACCURACY_THRESHOLDS_KM = (10, 50, 161)
