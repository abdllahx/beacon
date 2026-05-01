import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from beacon import db, imagery, vision

log = structlog.get_logger()


def run_verify(claim_id: int) -> dict:
    """Fetch before/after Sentinel-2 tiles for a geocoded claim and run Claude vision analysis.

    Persists everything to verification_runs.
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.raw_text, c.admin_region,
                       ST_XMin(c.bbox), ST_YMin(c.bbox), ST_XMax(c.bbox), ST_YMax(c.bbox),
                       a.published_at, a.title
                FROM claims c JOIN articles a ON a.id = c.article_id
                WHERE c.id = %s AND c.bbox IS NOT NULL
                """,
                (claim_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"status": "error", "reason": "claim_not_geocoded"}
        _, raw_text, admin_region, w, s, e, n, pub, title = row
        bbox = (w, s, e, n)
        event_date = pub or datetime.now(UTC)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO verification_runs (claim_id, status, started_at)
                VALUES (%s, 'running', now())
                RETURNING id
                """,
                (claim_id,),
            )
            run_id = cur.fetchone()[0]
        conn.commit()

        try:
            tiles = imagery.fetch_before_after_for_claim(
                claim_id=claim_id,
                bbox=bbox,
                event_date=event_date,
            )
            before_path = tiles["before"]["path"] if tiles.get("before") else None
            after_path = tiles["after"]["path"] if tiles.get("after") else None

            verdict = vision.analyze_tile_pair(
                claim_text=(title or "") + ("\n" + raw_text if raw_text else ""),
                place=admin_region or "",
                event_date=event_date.isoformat() if event_date else "",
                before_path=Path(before_path) if before_path else None,
                after_path=Path(after_path) if after_path else None,
                cwd=str(Path.cwd()),
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE verification_runs
                    SET status = 'vision_done',
                        before_tile_path = %s,
                        after_tile_path = %s,
                        vision_verdict = %s::jsonb,
                        completed_at = now()
                    WHERE id = %s
                    """,
                    (before_path, after_path, json.dumps(verdict), run_id),
                )
            conn.commit()

            return {
                "status": "ok",
                "run_id": run_id,
                "verdict": verdict.get("verdict"),
                "confidence": verdict.get("confidence"),
                "before": before_path,
                "after": after_path,
            }
        except Exception as e:
            log.exception("verify.error", claim_id=claim_id, run_id=run_id)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE verification_runs SET status='error', completed_at=now() WHERE id=%s",
                    (run_id,),
                )
            conn.commit()
            return {"status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
