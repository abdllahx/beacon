from operator import add
from typing import Annotated, TypedDict


class BeaconState(TypedDict, total=False):
    """State threaded through the Beacon verification DAG.

    All fields are optional at any point — nodes populate what they own, downstream
    nodes read what they need. Parallel branches (e.g. fetch_s2_before, fetch_s2_after,
    fetch_s1_before in Phase B) write disjoint keys.
    """

    # Inputs — supply either article_id (benchmark, full pipeline) or claim_id (demo, skip extract+geocode)
    article_id: int
    claim_id: int

    # Populated by load_claim
    article_title: str | None
    article_url: str | None
    raw_text: str | None
    bbox: tuple[float, float, float, float] | None
    admin_region: str | None
    event_date: str | None  # ISO 8601

    # Persistence handle
    run_id: int | None

    # Imagery (Phase A: Sentinel-2 visual; Phase B adds nbr, s1_*)
    s2_before: dict | None
    s2_after: dict | None

    # Phase B
    nbr_before: dict | None
    nbr_after: dict | None
    dnbr: dict | None
    # Phase B4 — SAR
    s1_before: dict | None
    s1_after: dict | None
    s1_change: dict | None
    # SegFormer land-cover segmentation result (HF Image Segmentation)
    segmentation: dict | None
    # DETR object detection result (HF Object Detection)
    detections: dict | None
    # SigLIP zero-shot tile classification (HF Zero-Shot Image Classification)
    tile_classification: dict | None
    # SigLIP-based archive retrieval matches (HF Visual Document Retrieval)
    vdr_matches: list | None

    # Vision/synth outputs
    vision_verdict: dict | None
    final_verdict: dict | None
    report_md: str | None

    # Phase C
    translations: dict | None
    article_summary: str | None

    # Cross-node error sink. Parallel branches may all append.
    errors: Annotated[list[dict], add]
