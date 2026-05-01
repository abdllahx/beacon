"""LangGraph node functions for the Beacon verification DAG.

Each node takes the full BeaconState and returns a partial dict — only the keys it
owns. LangGraph merges them. Nodes are deliberately thin wrappers around the existing
domain modules (`imagery`, `vision`, `synth`) so the M1 happy-path code keeps working
while the DAG provides orchestration, parallelism, and checkpointing.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from beacon import claude, db, imagery, vision
from beacon.graph.state import BeaconState

log = structlog.get_logger()


def extract_claim(state: BeaconState) -> dict:
    """Run NER + zero-shot on the article and create a claim row.

    No-op when the state already has a claim_id (demo path, or resumed run).
    """
    if state.get("claim_id"):
        return {}
    article_id = state.get("article_id")
    if not article_id:
        return {"errors": [{"node": "extract_claim", "msg": "neither article_id nor claim_id provided"}]}

    from huggingface_hub import InferenceClient

    from beacon.config import get_settings
    from beacon.extract import process_article

    settings = get_settings()
    if not settings.hf_token:
        return {"errors": [{"node": "extract_claim", "msg": "HF_TOKEN missing"}]}

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title, content FROM articles WHERE id = %s", (article_id,))
            row = cur.fetchone()
        if not row:
            return {"errors": [{"node": "extract_claim", "msg": f"article {article_id} not found"}]}
        title, content = row
        client = InferenceClient(token=settings.hf_token)
        try:
            result = process_article(conn, client, article_id, title, content)
        except Exception as e:
            log.exception("graph.extract_claim.error", article_id=article_id)
            return {"errors": [{"node": "extract_claim", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
        conn.commit()
        if result.get("status") != "kept":
            return {"errors": [{"node": "extract_claim", "msg": f"article rejected: {result}"}]}
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM claims WHERE article_id = %s ORDER BY id DESC LIMIT 1",
                (article_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"errors": [{"node": "extract_claim", "msg": "extract reported kept but no claim row"}]}
        claim_id = row[0]
    return {"claim_id": claim_id}


def geocode_claim_node(state: BeaconState) -> dict:
    """Run Nominatim + Claude disambiguation if the claim isn't yet geocoded.

    No-op when the claim already has status='geocoded' (demo path).
    """
    claim_id = state.get("claim_id")
    if not claim_id:
        return {}
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, locations, raw_text FROM claims WHERE id = %s", (claim_id,))
            row = cur.fetchone()
        if not row:
            return {"errors": [{"node": "geocode_claim", "msg": f"claim {claim_id} not found"}]}
        status, locations, raw_text = row
        if status == "geocoded":
            return {}
        from beacon.geocode import geocode_claim as _geocode

        try:
            result = _geocode(conn, claim_id, locations or [], raw_text or "")
        except Exception as e:
            log.exception("graph.geocode_claim.error", claim_id=claim_id)
            return {"errors": [{"node": "geocode_claim", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
        conn.commit()
        if result.get("status") != "geocoded":
            return {"errors": [{"node": "geocode_claim", "msg": f"geocoding failed: {result}"}]}
    return {}


def load_claim(state: BeaconState) -> dict:
    """Read claim row + linked article from Postgres, populate static context fields."""
    claim_id = state.get("claim_id")
    if not claim_id:
        return {"errors": [{"node": "load_claim", "msg": "no claim_id after extract+geocode"}]}
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.raw_text, c.admin_region,
                   ST_XMin(c.bbox), ST_YMin(c.bbox), ST_XMax(c.bbox), ST_YMax(c.bbox),
                   a.title, a.url, a.published_at
            FROM claims c JOIN articles a ON a.id = c.article_id
            WHERE c.id = %s AND c.bbox IS NOT NULL
            """,
            (claim_id,),
        )
        row = cur.fetchone()
    if not row:
        return {"errors": [{"node": "load_claim", "msg": f"claim {claim_id} not geocoded"}]}
    raw_text, admin_region, w, s, e, n, title, url, pub = row
    event_date = (pub or datetime.now(UTC)).isoformat()
    return {
        "raw_text": raw_text,
        "admin_region": admin_region,
        "bbox": (w, s, e, n),
        "article_title": title,
        "article_url": url,
        "event_date": event_date,
    }


def init_run(state: BeaconState) -> dict:
    """Open a verification_runs row so downstream nodes can persist intermediate output."""
    if state.get("bbox") is None:
        return {}
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO verification_runs (claim_id, status, started_at)
                VALUES (%s, 'running', now()) RETURNING id
                """,
                (state["claim_id"],),
            )
            run_id = cur.fetchone()[0]
        conn.commit()
    return {"run_id": run_id}


def fetch_s2_before(state: BeaconState) -> dict:
    if state.get("bbox") is None or state.get("event_date") is None:
        return {}
    bbox = tuple(state["bbox"])
    event_date = datetime.fromisoformat(state["event_date"])
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_before.png")
    try:
        info = imagery.fetch_tile(bbox, event_date, window_days=-30, out_path=out_path)
    except Exception as e:
        log.exception("graph.fetch_s2_before.error")
        return {"errors": [{"node": "fetch_s2_before", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"s2_before": info}


def fetch_s2_after(state: BeaconState) -> dict:
    if state.get("bbox") is None or state.get("event_date") is None:
        return {}
    bbox = tuple(state["bbox"])
    event_date = datetime.fromisoformat(state["event_date"])
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_after.png")
    try:
        info = imagery.fetch_tile(bbox, event_date, window_days=14, out_path=out_path)
    except Exception as e:
        log.exception("graph.fetch_s2_after.error")
        return {"errors": [{"node": "fetch_s2_after", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"s2_after": info}


def fetch_nbr_before(state: BeaconState) -> dict:
    if state.get("bbox") is None or state.get("event_date") is None:
        return {}
    bbox = tuple(state["bbox"])
    event_date = datetime.fromisoformat(state["event_date"])
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_nbr_before.png")
    try:
        info = imagery.fetch_nbr_tile(bbox, event_date, window_days=-30, out_path=out_path)
    except Exception as e:
        log.exception("graph.fetch_nbr_before.error")
        return {"errors": [{"node": "fetch_nbr_before", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"nbr_before": info}


def fetch_nbr_after(state: BeaconState) -> dict:
    if state.get("bbox") is None or state.get("event_date") is None:
        return {}
    bbox = tuple(state["bbox"])
    event_date = datetime.fromisoformat(state["event_date"])
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_nbr_after.png")
    try:
        info = imagery.fetch_nbr_tile(bbox, event_date, window_days=14, out_path=out_path)
    except Exception as e:
        log.exception("graph.fetch_nbr_after.error")
        return {"errors": [{"node": "fetch_nbr_after", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"nbr_after": info}


def compute_dnbr(state: BeaconState) -> dict:
    """Compute dNBR delta map from the before/after NBR float arrays."""
    nb = state.get("nbr_before") or {}
    na = state.get("nbr_after") or {}
    if not nb.get("array_path") or not na.get("array_path"):
        return {}
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_dnbr.png")
    try:
        info = imagery.compute_dnbr(
            Path(nb["array_path"]),
            Path(na["array_path"]),
            out_path,
        )
    except Exception as e:
        log.exception("graph.compute_dnbr.error")
        return {"errors": [{"node": "compute_dnbr", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"dnbr": info}


def fetch_s1_before(state: BeaconState) -> dict:
    if state.get("bbox") is None or state.get("event_date") is None:
        return {}
    bbox = tuple(state["bbox"])
    event_date = datetime.fromisoformat(state["event_date"])
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_s1_before.png")
    try:
        info = imagery.fetch_s1_tile(bbox, event_date, window_days=-30, out_path=out_path)
    except Exception as e:
        log.exception("graph.fetch_s1_before.error")
        return {"errors": [{"node": "fetch_s1_before", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"s1_before": info}


def fetch_s1_after(state: BeaconState) -> dict:
    if state.get("bbox") is None or state.get("event_date") is None:
        return {}
    bbox = tuple(state["bbox"])
    event_date = datetime.fromisoformat(state["event_date"])
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_s1_after.png")
    try:
        info = imagery.fetch_s1_tile(bbox, event_date, window_days=30, out_path=out_path)
    except Exception as e:
        log.exception("graph.fetch_s1_after.error")
        return {"errors": [{"node": "fetch_s1_after", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"s1_after": info}


def compute_s1_change(state: BeaconState) -> dict:
    s1b = state.get("s1_before") or {}
    s1a = state.get("s1_after") or {}
    if not s1b.get("array_path") or not s1a.get("array_path"):
        return {}
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_s1_change.png")
    try:
        info = imagery.compute_s1_change(
            Path(s1b["array_path"]),
            Path(s1a["array_path"]),
            out_path,
        )
    except Exception as e:
        log.exception("graph.compute_s1_change.error")
        return {"errors": [{"node": "compute_s1_change", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"s1_change": info}


def classify_tile(state: BeaconState) -> dict:
    """SigLIP zero-shot classification of the AFTER S2 tile against land/event labels.
    HF task: Zero-Shot Image Classification."""
    s2a = state.get("s2_after") or {}
    if not s2a.get("path"):
        return {}
    try:
        from beacon.siglip import zero_shot_classify

        ranked = zero_shot_classify(s2a["path"])
    except Exception as e:
        log.exception("graph.classify_tile.error")
        return {"errors": [{"node": "classify_tile", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {
        "tile_classification": {
            "model": "google/siglip-base-patch16-224",
            "ranked": ranked,
            "top_label": ranked[0]["label"] if ranked else None,
            "top_score": ranked[0]["score"] if ranked else None,
        }
    }


def vdr_search(state: BeaconState) -> dict:
    """SigLIP-based Visual Document Retrieval against the tile_archive.
    HF task: Visual Document Retrieval."""
    s2a = state.get("s2_after") or {}
    if not s2a.get("path"):
        return {}
    try:
        from beacon.vdr import search_archive

        matches = search_archive(s2a["path"], k=5)
    except Exception as e:
        log.exception("graph.vdr_search.error")
        return {"errors": [{"node": "vdr_search", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"vdr_matches": matches}


def detect_after(state: BeaconState) -> dict:
    """DETR object detection on the AFTER S2 tile. HF: Object Detection."""
    s2a = state.get("s2_after") or {}
    if not s2a.get("path"):
        return {}
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_detections.png")
    try:
        from beacon.detect import detect_objects

        info = detect_objects(s2a["path"], str(out_path))
    except Exception as e:
        log.exception("graph.detect_after.error")
        return {"errors": [{"node": "detect_after", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"detections": info}


def segment_after(state: BeaconState) -> dict:
    """SegFormer land-cover segmentation on the AFTER S2 tile. HF: Image Segmentation."""
    s2a = state.get("s2_after") or {}
    if not s2a.get("path"):
        return {}
    out_path = Path(f"data/tiles/claim_{state['claim_id']}_segmentation.png")
    try:
        from beacon.segment import segment_image

        info = segment_image(s2a["path"], str(out_path))
    except Exception as e:
        log.exception("graph.segment_after.error")
        return {"errors": [{"node": "segment_after", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"segmentation": info}


def vision_vqa(state: BeaconState) -> dict:
    title = state.get("article_title") or ""
    raw = state.get("raw_text") or ""
    claim_text = (title + ("\n" + raw if raw else ""))[:3000]
    before = state.get("s2_before") or {}
    after = state.get("s2_after") or {}
    dnbr = state.get("dnbr") or {}
    s1_change = state.get("s1_change") or {}
    try:
        verdict = vision.analyze_tile_pair(
            claim_text=claim_text,
            place=state.get("admin_region") or "",
            event_date=state.get("event_date") or "",
            before_path=Path(before["path"]) if before.get("path") else None,
            after_path=Path(after["path"]) if after.get("path") else None,
            dnbr_path=Path(dnbr["path"]) if dnbr.get("path") else None,
            dnbr_burn_pct=dnbr.get("burn_pct"),
            s1_change_path=Path(s1_change["path"]) if s1_change.get("path") else None,
            s1_decrease_pct=s1_change.get("decrease_pct"),
            cwd=str(Path.cwd()),
        )
    except Exception as e:
        log.exception("graph.vision_vqa.error")
        return {"errors": [{"node": "vision_vqa", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    return {"vision_verdict": verdict}


def persist_vision(state: BeaconState) -> dict:
    """Write the unified imagery_metadata JSONB + vision verdict to verification_runs."""
    run_id = state.get("run_id")
    if not run_id:
        return {}
    metadata: dict[str, dict] = {}
    if (s2b := state.get("s2_before")):
        metadata.setdefault("s2", {})["before"] = s2b
    if (s2a := state.get("s2_after")):
        metadata.setdefault("s2", {})["after"] = s2a
    if (nb := state.get("nbr_before")):
        metadata.setdefault("nbr", {})["before"] = nb
    if (na := state.get("nbr_after")):
        metadata.setdefault("nbr", {})["after"] = na
    if (dn := state.get("dnbr")):
        metadata.setdefault("nbr", {})["delta"] = dn
    if (s1b := state.get("s1_before")):
        metadata.setdefault("s1", {})["before"] = s1b
    if (s1a := state.get("s1_after")):
        metadata.setdefault("s1", {})["after"] = s1a
    if (s1c := state.get("s1_change")):
        metadata.setdefault("s1", {})["change"] = s1c
    if (seg := state.get("segmentation")):
        # SegFormer land-cover segmentation results, attached to the AFTER S2 node.
        metadata.setdefault("s2", {}).setdefault("after", {})
        if isinstance(metadata["s2"]["after"], dict):
            metadata["s2"]["after"]["segmentation"] = seg
    if (det := state.get("detections")):
        # DETR object detection results, attached to the AFTER S2 node.
        metadata.setdefault("s2", {}).setdefault("after", {})
        if isinstance(metadata["s2"]["after"], dict):
            metadata["s2"]["after"]["detections"] = det
    if (tc := state.get("tile_classification")):
        metadata.setdefault("s2", {}).setdefault("after", {})
        if isinstance(metadata["s2"]["after"], dict):
            metadata["s2"]["after"]["zero_shot"] = tc
    if (vdr := state.get("vdr_matches")):
        metadata["vdr_matches"] = vdr
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE verification_runs
            SET status = 'vision_done',
                imagery_metadata = %s::jsonb,
                vision_verdict = %s::jsonb
            WHERE id = %s
            """,
            (
                json.dumps(metadata),
                json.dumps(state.get("vision_verdict") or {}),
                run_id,
            ),
        )
        conn.commit()
    return {}


def synthesize(state: BeaconState) -> dict:
    """Reuse the M1 synthesizer, which reads the run row + FIRMS context from Postgres."""
    from beacon.synth import synthesize_run

    run_id = state.get("run_id")
    if not run_id:
        return {}
    try:
        res = synthesize_run(run_id)
    except Exception as e:
        log.exception("graph.synthesize.error")
        return {"errors": [{"node": "synthesize", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    if res.get("status") != "ok":
        return {"errors": [{"node": "synthesize", "msg": str(res)[:200]}]}
    return {
        "final_verdict": {
            "headline": res.get("headline"),
            "verdict": res.get("verdict"),
            "confidence": res.get("confidence"),
        },
        "report_md": res.get("report_markdown"),
    }


def summarize_article(state: BeaconState) -> dict:
    """Extractive summary of the article + claim via BART-CNN. HF task: Summarization."""
    run_id = state.get("run_id")
    title = state.get("article_title") or ""
    raw = state.get("raw_text") or ""
    text = (title + "\n\n" + raw).strip()
    if not text or not run_id:
        return {}
    try:
        from beacon.summarize import summarize as bart_summarize

        summary = bart_summarize(text)
    except Exception as e:
        log.exception("graph.summarize_article.error")
        return {"errors": [{"node": "summarize_article", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    if summary:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE verification_runs SET article_summary = %s WHERE id = %s",
                (summary, run_id),
            )
            conn.commit()
    return {"article_summary": summary}


def translate_report(state: BeaconState) -> dict:
    """Translate the synthesized report into target languages via Helsinki-NLP MarianMT.

    HF task #6 — Translation. Persists into verification_runs.translations JSONB so
    the Streamlit UI's language toggle can read them without re-translating.
    """
    report_md = state.get("report_md")
    run_id = state.get("run_id")
    if not report_md or not run_id:
        return {}
    try:
        from beacon.translate import translate_all

        translations = translate_all(report_md)
    except Exception as e:
        log.exception("graph.translate_report.error")
        return {"errors": [{"node": "translate_report", "msg": f"{type(e).__name__}: {e!s:.160}"}]}
    # Filter out failed translations (None values) before persisting
    clean = {k: v for k, v in translations.items() if v}
    if clean:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE verification_runs SET translations = %s::jsonb WHERE id = %s",
                (json.dumps(clean), run_id),
            )
            conn.commit()
    return {"translations": clean}


# Used by claude to silence its unused-import warning when callers reach into beacon.claude
_ = claude
