import json

import structlog

from beacon import claude, db

log = structlog.get_logger()


SYSTEM_PROMPT = """You are a verification analyst writing a concise, evidence-backed report on a
news claim that was cross-referenced against satellite imagery and (where available) ground-truth
fire detection data.

Output a single JSON object with these keys:

{
  "headline": "one-line summary verdict",
  "verdict": "supported" | "refuted" | "inconclusive",
  "confidence": 0.0,
  "report_markdown": "1-2 paragraphs of grounded prose synthesizing the claim, the imagery findings, and the ground-truth correlation, written for a journalist or analyst audience. Cite the Sentinel-2 imagery and FIRMS data explicitly. Be precise about what is established vs uncertain."
}

The report must be honest about uncertainty. Do not embellish. Reference specific evidence from
the inputs (imagery findings, FIRMS detection counts, locations). No prose outside the JSON."""


def _summarize_firms_for_claim(conn, claim_id: int, days_window: int = 14) -> dict:
    """Return aggregate FIRMS hits within the claim's bbox, +/- days_window of the article date."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH ctx AS (
                SELECT c.bbox, a.published_at
                FROM claims c JOIN articles a ON a.id = c.article_id
                WHERE c.id = %s
            )
            SELECT
                count(*) AS n_hits,
                round(avg(f.frp)::numeric, 2) AS mean_frp,
                round(max(f.frp)::numeric, 2) AS max_frp,
                min(f.detected_at) AS earliest,
                max(f.detected_at) AS latest
            FROM firms_events f, ctx
            WHERE ST_Within(f.point, ctx.bbox)
              AND (ctx.published_at IS NULL
                   OR f.detected_at BETWEEN ctx.published_at - make_interval(days => %s)
                                        AND ctx.published_at + make_interval(days => %s))
            """,
            (claim_id, days_window, days_window),
        )
        row = cur.fetchone()
    n, mean_frp, max_frp, earliest, latest = row
    return {
        "n_hits": int(n or 0),
        "mean_frp_mw": float(mean_frp) if mean_frp is not None else None,
        "max_frp_mw": float(max_frp) if max_frp is not None else None,
        "earliest": earliest.isoformat() if earliest else None,
        "latest": latest.isoformat() if latest else None,
    }


def synthesize_run(run_id: int) -> dict:
    """Read a verification_run + linked claim, ask Claude for a final verdict + markdown report."""
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT v.id, v.claim_id, v.imagery_metadata, v.vision_verdict,
                       c.raw_text, c.admin_region, c.event_type,
                       a.title, a.url, a.source, a.published_at
                FROM verification_runs v
                JOIN claims c ON c.id = v.claim_id
                JOIN articles a ON a.id = c.article_id
                WHERE v.id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"status": "error", "reason": "run_not_found"}
        (
            _,
            claim_id,
            imagery_metadata,
            vision_verdict,
            raw_text,
            admin_region,
            event_type,
            title,
            url,
            source,
            published_at,
        ) = row
        firms_summary = _summarize_firms_for_claim(conn, claim_id)

    inputs = {
        "article": {
            "title": title,
            "url": url,
            "source": source,
            "published_at": published_at.isoformat() if published_at else None,
        },
        "claim": {
            "raw_text": (raw_text or "")[:1500],
            "event_type": event_type,
            "admin_region": admin_region,
        },
        "imagery": imagery_metadata or {},
        "vision_verdict": vision_verdict,
        "firms_ground_truth": firms_summary,
    }

    prompt = (
        "Synthesize a final verification report from the following evidence.\n\n"
        f"{json.dumps(inputs, indent=2, default=str)}\n\n"
        "Output the JSON object as specified. No preamble."
    )

    raw = claude.ask(prompt, system_prompt=SYSTEM_PROMPT, max_turns=1)
    parsed = claude.parse_json_block(raw)
    if not parsed:
        log.warning("synth.unparseable", run_id=run_id, raw_preview=raw[:300])
        parsed = {
            "headline": "synthesis failed",
            "verdict": "inconclusive",
            "confidence": 0.0,
            "report_markdown": "Synthesis output could not be parsed as JSON.",
        }

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE verification_runs
            SET status = 'synth_done',
                final_verdict = %s::jsonb,
                final_report_md = %s,
                completed_at = now()
            WHERE id = %s
            """,
            (json.dumps(parsed), parsed.get("report_markdown", ""), run_id),
        )
        conn.commit()

    return {"status": "ok", "run_id": run_id, **parsed, "firms": firms_summary}
