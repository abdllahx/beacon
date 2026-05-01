"""500-event benchmark builder + runner.

Sample EM-DAT events with valid bbox + Sentinel-2-era dates, synthesize a claim per
event, run the LangGraph verification pipeline against each. Persist outcomes to
`benchmark_runs` for the metrics module.

Each EM-DAT event is taken as ground truth that the disaster occurred — so the
expected verdict is "supported". Recall on supported is the headline metric.
"""

from __future__ import annotations

import hashlib
from datetime import date

import structlog

from beacon import db
from beacon.graph import run_pipeline

log = structlog.get_logger()

# v2 methodology: insert ONLY an article and let the LangGraph pipeline run extract +
# geocode from text alone. This makes IoU and centroid distance vs EM-DAT ground truth
# meaningful (v1 cheated by injecting EM-DAT bbox directly into the claim).
BENCHMARK_SOURCE = "emdat-benchmark-v2"


def _synthesize_claim_text(event: dict) -> str:
    """Build news-like synthetic article text from an EM-DAT row.

    Strategy: put the MOST SPECIFIC place name first (in the title, where NER weighs
    heaviest) and bury the country in the body as context. EM-DAT's `location` field
    often has parentheticals like "Siskiyou County (California)" — strip them so NER
    sees clean entity strings.
    """
    dt = event.get("disaster_type") or "Disaster"
    country = (event.get("country") or "").strip()
    location = (event.get("location") or "").strip()

    # Pull the leading-most-specific place name (before any "(", ",", or ";")
    head = location.split("(")[0].split(",")[0].split(";")[0].strip(" .'\"")
    head = head[:80]
    parenthetical = location[len(head):].lstrip(" ,;").strip(" .'\"()")[:160]

    if head:
        title = f"{dt} devastates {head}"
    elif country:
        title = f"{dt} hits {country}"
    else:
        title = f"{dt} reported"

    body_parts: list[str] = []
    body_parts.append(f"A {dt.lower()} struck {head or country}")
    if parenthetical:
        body_parts.append(f"({parenthetical})")
    if country and country.lower() not in (head or "").lower():
        body_parts.append(f"in {country}")
    if start := event.get("start_date"):
        body_parts.append(f"on {start.strftime('%B %d, %Y')}")
    if (deaths := event.get("total_deaths")) and deaths > 0:
        body_parts.append(f"killing at least {int(deaths)} people")
    if (affected := event.get("total_affected")) and affected > 0:
        body_parts.append(f"and affecting an estimated {int(affected):,} residents")
    body = " ".join(body_parts) + "."
    return f"{title}\n\n{body}"


def _synthesize_article(conn, event: dict) -> int:
    """Insert (or fetch) a synthetic news-style article for an EM-DAT event.

    Critically, this does NOT create a claim row — the LangGraph pipeline will run
    extract + geocode against this article and reconstruct the claim from text alone.
    """
    text = _synthesize_claim_text(event)
    title_part = text.split("\n\n", 1)[0]
    url = f"emdat://{event['dis_no']}-v2"
    url_h = hashlib.sha256(url.encode()).hexdigest()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO articles (source, url, url_hash, title, content, language, published_at, extract_status)
            VALUES (%s, %s, %s, %s, %s, 'en', %s, NULL)
            ON CONFLICT (url_hash) DO UPDATE SET extract_status = NULL
            RETURNING id
            """,
            (BENCHMARK_SOURCE, url, url_h, title_part, text, event["start_date"]),
        )
        return cur.fetchone()[0]


def sample_events(
    n: int,
    *,
    types: tuple[str, ...] = ("Wildfire", "Flood", "Storm", "Earthquake"),
    min_year: int = 2018,
    max_year: int = 2025,
    seed: int | None = None,
    gdis_only: bool = False,
    exclude_existing: bool = True,
) -> list[dict]:
    """Stratified random sample of EM-DAT events. We still require bbox IS NOT NULL on
    the EM-DAT side because we need ground truth to score against — but the bbox is
    NOT passed to the pipeline; only article text is.

    Set `gdis_only=True` to sample only events with GDIS peer-reviewed ground truth
    (the rigorous benchmark subset). Note GDIS doesn't cover wildfires.

    Set `exclude_existing=False` to allow re-sampling events already registered in
    benchmark_runs (default excludes them so growing N=7 → N=50 adds *new* events).
    """
    per_type = max(1, n // len(types))
    out: list[dict] = []
    # v2 methodology: pipeline reconstructs bbox from article text, so we don't need
    # emdat_events.bbox at sample time. We DO need ground truth — that comes from
    # disaster_ground_truth (GDIS preferred, EM-DAT fallback). For gdis_only=True
    # we further constrain to GDIS-covered events.
    if gdis_only:
        gt_clause = """
            JOIN disaster_ground_truth dgt ON dgt.emdat_event_id = e.id AND dgt.gt_source = 'gdis'
        """
    else:
        gt_clause = """
            JOIN disaster_ground_truth dgt ON dgt.emdat_event_id = e.id
        """
    exclude_clause = (
        "AND NOT EXISTS (SELECT 1 FROM benchmark_runs br WHERE br.emdat_event_id = e.id)"
        if exclude_existing
        else ""
    )
    with db.connect() as conn, conn.cursor() as cur:
        if seed is not None:
            cur.execute("SELECT setseed(%s)", (max(-1.0, min(1.0, seed / 1e9)),))
        for t in types:
            cur.execute(
                f"""
                SELECT id, dis_no, disaster_type, country, location,
                       start_date, total_deaths, total_affected
                FROM (
                    SELECT DISTINCT e.id, e.dis_no, e.disaster_type, e.country, e.location,
                           e.start_date, e.total_deaths, e.total_affected
                    FROM emdat_events e
                    {gt_clause}
                    WHERE dgt.gt_bbox IS NOT NULL
                      AND e.start_date BETWEEN %s AND %s
                      AND e.disaster_type = %s
                      {exclude_clause}
                ) sub
                ORDER BY random()
                LIMIT %s
                """,
                (date(min_year, 1, 1), date(max_year, 12, 31), t, per_type),
            )
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                out.append(dict(zip(cols, row, strict=False)))
    return out


def build_benchmark(
    n: int = 50,
    *,
    types: tuple[str, ...] = ("Wildfire", "Flood", "Storm", "Earthquake"),
    gdis_only: bool = False,
) -> dict[str, int]:
    """v2 — insert article rows only and register them for benchmarking. The pipeline
    will run extract+geocode from text alone, producing an independently reconstructed
    bbox that we can fairly compare to ground truth (GDIS or EM-DAT)."""
    events = sample_events(n, types=types, gdis_only=gdis_only)
    counts = {"requested": n, "sampled": len(events), "articles_built": 0, "registered": 0}
    with db.connect() as conn:
        for ev in events:
            article_id = _synthesize_article(conn, ev)
            counts["articles_built"] += 1
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO benchmark_runs (emdat_event_id, article_id, expected_verdict)
                    VALUES (%s, %s, 'supported')
                    ON CONFLICT (emdat_event_id) DO UPDATE SET article_id = EXCLUDED.article_id
                    RETURNING (xmax = 0) AS inserted
                    """,
                    (ev["id"], article_id),
                )
                inserted = cur.fetchone()[0]
                if inserted:
                    counts["registered"] += 1
        conn.commit()
    log.info("benchmark.build_done", **counts)
    return counts


def run_benchmark(limit: int | None = None, *, gdis_only_metrics: bool = True) -> dict[str, int]:
    """Execute the full LangGraph pipeline on every benchmark article that hasn't run yet.

    After each event, prints a `PROGRESS` line with rolling metrics (Acc@50km, mean
    centroid distance, recall on supported) so a long-running benchmark is observable
    in real time. If the metrics move in a bad direction, the run can be killed early.
    """
    from beacon.eval_metrics import compute_metrics

    counts = {"ok": 0, "errors": 0, "skipped": 0}
    with db.connect() as conn, conn.cursor() as cur:
        sql = """
            SELECT br.id, br.emdat_event_id, br.article_id, ee.dis_no, ee.disaster_type
            FROM benchmark_runs br
            JOIN emdat_events ee ON ee.id = br.emdat_event_id
            WHERE br.beacon_run_id IS NULL AND br.article_id IS NOT NULL
            ORDER BY br.id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        cur.execute(sql)
        rows = cur.fetchall()
    total = len(rows)
    for idx, (br_id, _emdat_id, article_id, dis_no, dtype) in enumerate(rows, start=1):
        log.info("benchmark.run.start", idx=idx, total=total, br_id=br_id, dis_no=dis_no, type=dtype)
        try:
            state = run_pipeline(article_id=article_id, thread_id=f"bench-v2-{br_id}")
            run_id = state.get("run_id")
            claim_id = state.get("claim_id")
            verdict = (state.get("final_verdict") or {}).get("verdict")
            confidence = (state.get("final_verdict") or {}).get("confidence")
            errs = state.get("errors") or []
            notes = "; ".join(e.get("msg", "") for e in errs) if errs else None
            with db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE benchmark_runs
                    SET beacon_run_id = %s,
                        claim_id = %s,
                        beacon_verdict = %s,
                        beacon_confidence = %s,
                        notes = %s
                    WHERE id = %s
                    """,
                    (run_id, claim_id, verdict, confidence, notes, br_id),
                )
                conn.commit()
            counts["ok"] += 1
        except Exception as e:
            log.exception("benchmark.run.error", br_id=br_id)
            counts["errors"] += 1
            verdict, confidence = "error", None

        # Rolling metrics after every event so we can pause if numbers go bad.
        try:
            m = compute_metrics(gdis_only=gdis_only_metrics)
            geo = (m.get("geoparsing") or {})
            acc = (geo.get("accuracy") or {})
            print(
                f"PROGRESS [{idx}/{total}] dis_no={dis_no} type={dtype} "
                f"verdict={verdict} conf={confidence} | "
                f"rolling: n={m.get('n')} "
                f"Acc@10km={acc.get('@10km')} "
                f"Acc@50km={acc.get('@50km')} "
                f"Acc@161km={acc.get('@161km')} "
                f"mean_dist_km={geo.get('mean_centroid_distance_km')} "
                f"recall_supported={m.get('recall_supported')} "
                f"inconclusive={m.get('inconclusive_rate')}",
                flush=True,
            )
        except Exception:
            log.exception("benchmark.rolling_metrics.failed", br_id=br_id)
    return counts
