import typer

app = typer.Typer(help="Beacon — multimodal geospatial event verification.")


@app.command()
def health() -> None:
    """Check that config loads and Postgres is reachable."""
    from beacon import db
    from beacon.config import get_settings

    settings = get_settings()
    if settings.database_url:
        from urllib.parse import urlparse
        parsed = urlparse(settings.database_url)
        typer.echo(f"postgres -> {parsed.hostname}{parsed.path} (DATABASE_URL)")
    else:
        typer.echo(f"postgres -> {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT version(), postgis_version()")
        version, postgis = cur.fetchone()
    typer.echo(f"connected: {version.splitlines()[0]}")
    typer.echo(f"postgis:   {postgis}")


@app.command()
def ingest(
    query: str = typer.Option(None, help="Override the default wildfire query"),
    hours: int = typer.Option(48, "--hours", help="Hours back to fetch (NewsAPI dev tier has ~24h delay so 48h+ recommended)"),
    limit: int = typer.Option(50, "--limit", help="Max articles per source"),
    sources: str = typer.Option("gdelt,newsapi", help="Comma-separated source list"),
) -> None:
    """Fetch news articles into the articles table."""
    from beacon.ingest import DEFAULT_WILDFIRE_QUERY, run_ingest

    q = query or DEFAULT_WILDFIRE_QUERY
    src_tuple = tuple(s.strip() for s in sources.split(",") if s.strip())
    counts = run_ingest(q, hours_back=hours, max_per_source=limit, sources=src_tuple)
    for k, v in counts.items():
        typer.echo(f"{k:20s} = {v}")


@app.command()
def extract(
    limit: int = typer.Option(None, "--limit", help="Max articles to process this run"),
) -> None:
    """Run claim extraction (NER + zero-shot relevance) on unprocessed articles."""
    from beacon.extract import run_extract

    counts = run_extract(limit=limit)
    for k, v in counts.items():
        typer.echo(f"{k:12s} = {v}")


@app.command()
def geocode(
    limit: int = typer.Option(None, "--limit", help="Max claims to geocode this run"),
) -> None:
    """Resolve claim locations to bounding boxes via Nominatim."""
    from beacon.geocode import run_geocode

    counts = run_geocode(limit=limit)
    for k, v in counts.items():
        typer.echo(f"{k:14s} = {v}")


@app.command("firms-load")
def firms_load(
    area: str = typer.Option("-141,48,-115,60", help="west,south,east,north (default: British Columbia)"),
    days: int = typer.Option(5, help="Day range, 1-5 (FIRMS API limit)"),
    date: str = typer.Option(None, help="YYYY-MM-DD; omit for latest available"),
    source: str = typer.Option("VIIRS_SNPP_NRT", help="FIRMS product"),
) -> None:
    """Load NASA FIRMS fire detections for an area into firms_events."""
    from datetime import date as dt

    from beacon.firms import run_firms_load

    parts = [float(x) for x in area.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("area must be west,south,east,north")
    d = dt.fromisoformat(date) if date else None
    counts = run_firms_load(area=tuple(parts), days=days, date=d, source=source)
    for k, v in counts.items():
        typer.echo(f"{k:12s} = {v}")


@app.command("fetch-tiles")
def fetch_tiles(
    claim_id: int = typer.Argument(..., help="claims.id to fetch tiles for"),
    before_days: int = typer.Option(30, help="Days before event to search for 'before' tile"),
    after_days: int = typer.Option(14, help="Days after event to search for 'after' tile"),
    size_px: int = typer.Option(1024, help="Output image size on the long edge"),
) -> None:
    """Fetch before/after Sentinel-2 visual tiles for a geocoded claim."""
    from datetime import UTC, datetime

    from beacon import db, imagery

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, ST_XMin(c.bbox), ST_YMin(c.bbox), ST_XMax(c.bbox), ST_YMax(c.bbox),
                   a.published_at
            FROM claims c JOIN articles a ON a.id = c.article_id
            WHERE c.id = %s AND c.bbox IS NOT NULL
            """,
            (claim_id,),
        )
        row = cur.fetchone()
    if not row:
        raise typer.BadParameter(f"claim {claim_id} not geocoded or not found")
    _, w, s, e, n, pub = row
    event_date = pub or datetime.now(UTC)
    res = imagery.fetch_before_after_for_claim(
        claim_id=claim_id,
        bbox=(w, s, e, n),
        event_date=event_date,
        before_days=before_days,
        after_days=after_days,
        size_px=size_px,
    )
    typer.echo(f"event_date: {event_date.isoformat()}")
    for k, v in res.items():
        typer.echo(f"--- {k} ---")
        typer.echo(v if v else "  (no item found)")


@app.command()
def verify(
    claim_id: int = typer.Argument(..., help="claims.id of a geocoded claim"),
) -> None:
    """Fetch tiles + run Claude vision analysis for a geocoded claim."""
    import json as _json

    from beacon.verify import run_verify

    res = run_verify(claim_id)
    typer.echo(_json.dumps(res, indent=2, default=str))


@app.command()
def synthesize(
    run_id: int = typer.Argument(..., help="verification_runs.id"),
) -> None:
    """Produce final markdown report + verdict for a completed vision run."""
    import json as _json

    from beacon.synth import synthesize_run

    res = synthesize_run(run_id)
    typer.echo(_json.dumps(res, indent=2, default=str))


@app.command("graph-run")
def graph_run(
    claim_id: int = typer.Argument(..., help="claims.id of a geocoded claim"),
    thread_id: str = typer.Option(None, "--thread-id", help="Resume a previous run by passing the same thread_id"),
) -> None:
    """Run the LangGraph verification DAG against a claim (with SQLite checkpointing)."""
    import json as _json

    from beacon.graph import run_pipeline

    state = run_pipeline(claim_id, thread_id=thread_id)
    summary = {
        "thread_id": state.get("_thread_id"),
        "run_id": state.get("run_id"),
        "verdict": (state.get("final_verdict") or {}).get("verdict"),
        "confidence": (state.get("final_verdict") or {}).get("confidence"),
        "headline": (state.get("final_verdict") or {}).get("headline"),
        "errors": state.get("errors") or [],
    }
    typer.echo(_json.dumps(summary, indent=2, default=str))


@app.command("graph-render")
def graph_render(
    out: str = typer.Option(None, "--out", help="Write Mermaid to a file instead of stdout"),
) -> None:
    """Print the LangGraph DAG as a Mermaid diagram."""
    from beacon.graph import render_mermaid

    text = render_mermaid()
    if out:
        from pathlib import Path

        Path(out).write_text(text)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


@app.command()
def ui(port: int = typer.Option(8501, "--port")) -> None:
    """Launch the Streamlit verification dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).resolve().parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=False,
    )


@app.command("gdis-load")
def gdis_load_cmd(
    file: str = typer.Option("data/raw/gdis.csv", "--file", help="Path to GDIS disaster locations CSV"),
) -> None:
    """Load the GDIS peer-reviewed geocoded disaster dataset (Rosvold & Buhaug 2021)."""
    from pathlib import Path

    from beacon.gdis_loader import load_gdis

    counts = load_gdis(Path(file))
    for k, v in counts.items():
        typer.echo(f"{k:24s} = {v}")


@app.command("acled-login")
def acled_login_cmd(
    username: str = typer.Option(None, "--username", help="myACLED email; falls back to .env ACLED_USERNAME"),
    password: str = typer.Option(None, "--password", help="myACLED password; falls back to .env ACLED_PASSWORD"),
) -> None:
    """One-time ACLED OAuth login. Persists tokens (not password) to .acled_tokens.json."""
    from beacon.acled import login
    from beacon.config import get_settings

    settings = get_settings()
    user = username or settings.acled_username
    pw = password or settings.acled_password
    if not user or not pw:
        raise typer.BadParameter(
            "Need myACLED credentials. Pass --username / --password, or set "
            "ACLED_USERNAME and ACLED_PASSWORD in .env."
        )
    tokens = login(user, pw)
    typer.echo(f"login OK · access_token expires in {tokens.get('expires_in')}s")


@app.command("acled-load")
def acled_load_cmd(
    countries: str = typer.Option("Ukraine,Israel,Lebanon,Syria,Yemen,Sudan,Myanmar", "--countries"),
    year_start: int = typer.Option(2018, "--year-start"),
    year_end: int = typer.Option(2025, "--year-end"),
    types: str = typer.Option("Battles,Explosions/Remote violence", "--types"),
) -> None:
    """Load ACLED conflict events for (countries × years × event types) into acled_events."""
    from beacon.acled import load_country_years

    country_list = [c.strip() for c in countries.split(",") if c.strip()]
    year_list = list(range(int(year_start), int(year_end) + 1))
    type_tuple = tuple(t.strip() for t in types.split(",") if t.strip())
    counts = load_country_years(country_list, year_list, event_types=type_tuple)
    for k, v in counts.items():
        typer.echo(f"{k:12s} = {v}")


@app.command("emdat-load")
def emdat_load_cmd(
    file: str = typer.Option("data/raw/emdat.xlsx", "--file", help="Path to EM-DAT public xlsx export"),
) -> None:
    """Load EM-DAT events (wildfires/floods/earthquakes/storms) into emdat_events."""
    from pathlib import Path

    from beacon.emdat_loader import load_emdat

    counts = load_emdat(Path(file))
    for k, v in counts.items():
        typer.echo(f"{k:12s} = {v}")


@app.command("eval-build")
def eval_build_cmd(
    n: int = typer.Option(50, "--n", help="Number of events to sample"),
    types: str = typer.Option(
        "Wildfire,Flood,Storm,Earthquake",
        "--types",
        help="Comma-separated EM-DAT Disaster Type filter",
    ),
    gdis_only: bool = typer.Option(
        False, "--gdis-only/--all-events",
        help="Sample only events with GDIS peer-reviewed ground truth (excludes wildfires)",
    ),
) -> None:
    """Sample N EM-DAT events and stage them as benchmark claims."""
    from beacon.benchmark import build_benchmark

    type_tuple = tuple(t.strip() for t in types.split(",") if t.strip())
    counts = build_benchmark(n, types=type_tuple, gdis_only=gdis_only)
    for k, v in counts.items():
        typer.echo(f"{k:18s} = {v}")


@app.command("eval-run")
def eval_run_cmd(
    limit: int = typer.Option(None, "--limit", help="Max benchmark claims to run this invocation"),
) -> None:
    """Execute the LangGraph pipeline against staged benchmark claims."""
    from beacon.benchmark import run_benchmark

    counts = run_benchmark(limit=limit)
    for k, v in counts.items():
        typer.echo(f"{k:10s} = {v}")


@app.command("vdr-seed")
def vdr_seed_cmd() -> None:
    """Seed the tile_archive with the existing demo wildfire AFTER tiles."""
    from beacon.vdr import seed_demo_archive

    counts = seed_demo_archive()
    for k, v in counts.items():
        typer.echo(f"{k:8s} = {v}")


@app.command("vdr-search")
def vdr_search_cmd(
    image_path: str = typer.Argument(..., help="Path to a query image"),
    k: int = typer.Option(5, "-k"),
) -> None:
    """Visual Document Retrieval: top-K archived tiles most similar to the query."""
    import json as _json

    from beacon.vdr import search_archive

    typer.echo(_json.dumps(search_archive(image_path, k=k), indent=2, default=str))


@app.command("vdr-by-text")
def vdr_by_text_cmd(
    query: str = typer.Argument(..., help="Text query (cross-modal SigLIP retrieval)"),
    k: int = typer.Option(5, "-k"),
) -> None:
    """Cross-modal SigLIP retrieval: find archived tiles that match a text description."""
    import json as _json

    from beacon.vdr import search_by_text

    typer.echo(_json.dumps(search_by_text(query, k=k), indent=2, default=str))


@app.command("translate-run")
def translate_run_cmd(
    run_id: int = typer.Argument(..., help="verification_runs.id"),
    langs: str = typer.Option("es,ar,fr", "--langs", help="Comma-separated language codes"),
) -> None:
    """Translate an existing run's report into target languages (HF task: Translation)."""
    import json as _json

    from beacon import db
    from beacon.translate import translate_all

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT final_report_md FROM verification_runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            typer.echo(f"run {run_id} has no final_report_md")
            raise typer.Exit(1)
        text = row[0]
    target_langs = tuple(L.strip() for L in langs.split(",") if L.strip())
    translations = translate_all(text, target_langs=target_langs)
    clean = {k: v for k, v in translations.items() if v}
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE verification_runs SET translations = %s::jsonb WHERE id = %s",
            (_json.dumps(clean), run_id),
        )
        conn.commit()
    for lang, snippet in clean.items():
        typer.echo(f"--- {lang} ({len(snippet)} chars) ---")
        typer.echo(snippet[:300])
        typer.echo("")


@app.command("embed-claims")
def embed_claims_cmd(
    limit: int = typer.Option(None, "--limit"),
    all: bool = typer.Option(False, "--all", help="Re-embed even claims that already have an embedding"),
) -> None:
    """Compute sentence-transformers embeddings for claims (HF task: Sentence Similarity)."""
    from beacon.embed import run_embed_backfill

    counts = run_embed_backfill(limit=limit, only_missing=not all)
    for k, v in counts.items():
        typer.echo(f"{k:14s} = {v}")


@app.command("dedup")
def dedup_cmd(
    threshold: float = typer.Option(0.85, "--threshold", help="Cosine similarity threshold"),
    limit: int = typer.Option(20, "--limit"),
    all_types: bool = typer.Option(False, "--all-types", help="Compare across event types too"),
) -> None:
    """Find near-duplicate claims via cosine similarity (sentence-transformers)."""
    import json as _json

    from beacon.embed import find_near_duplicates

    pairs = find_near_duplicates(threshold=threshold, limit=limit, same_event_type_only=not all_types)
    typer.echo(_json.dumps(pairs, indent=2, default=str))


@app.command("similar")
def similar_cmd(
    query: str = typer.Argument(..., help="Free-text query"),
    k: int = typer.Option(5, "-k"),
) -> None:
    """Top-K most similar existing claims to a query string."""
    import json as _json

    from beacon.embed import search_similar

    typer.echo(_json.dumps(search_similar(query, k=k), indent=2, default=str))


@app.command("eval-progress")
def eval_progress_cmd(
    gdis_only: bool = typer.Option(True, "--gdis-only/--all"),
) -> None:
    """Quick rolling-metrics check while a benchmark is in flight."""
    import json as _json

    from beacon import db
    from beacon.eval_metrics import compute_metrics

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE beacon_run_id IS NOT NULL) AS done,
                   count(*) FILTER (WHERE beacon_run_id IS NULL AND article_id IS NOT NULL) AS pending,
                   count(*) AS total
            FROM benchmark_runs
            """
        )
        done, pending, total = cur.fetchone()
    typer.echo(f"events: done={done} pending={pending} total={total}")
    typer.echo(_json.dumps(compute_metrics(gdis_only=gdis_only), indent=2, default=str))


@app.command("eval-snapshot")
def eval_snapshot_cmd(
    label: str = typer.Argument(..., help="Short label for this snapshot, e.g. 'v2-N7-baseline'"),
    gdis_only: bool = typer.Option(False, "--gdis-only"),
) -> None:
    """Capture a versioned eval snapshot to data/eval_snapshots/."""
    from beacon.snapshots import capture

    path = capture(label, gdis_only=gdis_only)
    typer.echo(f"wrote {path}")


@app.command("eval-snapshots")
def eval_snapshots_cmd() -> None:
    """List all eval snapshots."""
    import json as _json

    from beacon.snapshots import list_snapshots

    typer.echo(_json.dumps(list_snapshots(), indent=2, default=str))


@app.command("eval-diff")
def eval_diff_cmd(
    a: str = typer.Argument(..., help="Earlier snapshot id (substring match)"),
    b: str = typer.Argument(..., help="Later snapshot id"),
) -> None:
    """Diff two eval snapshots: metric deltas + per-event verdict changes."""
    import json as _json

    from beacon.snapshots import diff

    typer.echo(_json.dumps(diff(a, b), indent=2, default=str))


@app.command("eval-report")
def eval_report_cmd(
    gdis_only: bool = typer.Option(
        False, "--gdis-only", help="Restrict to the GDIS peer-reviewed subset"
    ),
) -> None:
    """Aggregate verdicts + Accuracy@N km + IoU across completed benchmark runs."""
    import json as _json

    from beacon.eval_metrics import compute_metrics

    typer.echo(_json.dumps(compute_metrics(gdis_only=gdis_only), indent=2, default=str))


@app.command("emdat-geocode")
def emdat_geocode_cmd(
    limit: int = typer.Option(None, "--limit", help="Max events to geocode this run"),
    types: str = typer.Option(
        "Wildfire,Flood,Storm,Earthquake",
        "--types",
        help="Comma-separated EM-DAT Disaster Type filter",
    ),
) -> None:
    """Geocode EM-DAT events using Admin Units JSON via Nominatim."""
    from beacon.emdat_geocoder import populate_native_bbox, run_admin_geocoder

    native = populate_native_bbox()
    typer.echo(f"native_bboxes_set = {native}")
    type_tuple = tuple(t.strip() for t in types.split(",") if t.strip())
    counts = run_admin_geocoder(limit=limit, only_event_types=type_tuple)
    for k, v in counts.items():
        typer.echo(f"{k:20s} = {v}")


@app.command("demo-seed")
def demo_seed_cmd() -> None:
    """Seed 5 hand-curated historical wildfire events as geocoded claims."""
    from beacon.demo_seed import seed_all

    ids = seed_all()
    typer.echo(f"seeded claim_ids: {ids}")


@app.command("latency-report")
def latency_report() -> None:
    """Compute p50/p95/p99 wall-clock latency over verification_runs."""
    import json as _json

    from beacon.latency import latency_stats

    typer.echo(_json.dumps(latency_stats(), indent=2))


@app.command("cost-report")
def cost_report() -> None:
    """Aggregate cost_events: per-operation calls, USD, mean latency."""
    from beacon.cost import aggregate_json

    typer.echo(aggregate_json())


@app.command("cost-backfill")
def cost_backfill_cmd() -> None:
    """Estimate cost_events for past verification_runs (char-count heuristic).

    Operations are tagged '_estimated' so they're distinguishable from live-logged
    calls. Idempotent — only runs missing cost_events get backfilled."""
    import json as _json

    from beacon.cost import backfill_from_runs

    typer.echo(_json.dumps(backfill_from_runs(), indent=2))


@app.command("dspy-status")
def dspy_status_cmd() -> None:
    """Report DSPy install state + number of demos available for few-shot bootstrap."""
    import json as _json

    from beacon.dspy_synth import status

    typer.echo(_json.dumps(status(), indent=2))


@app.command("feedback-export")
def feedback_export(
    out: str = typer.Option("data/feedback.jsonl", "--out", help="Output JSONL path"),
) -> None:
    """Dump HITL feedback rows for offline review / DSPy training set."""
    import json as _json
    from pathlib import Path

    from beacon import db

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, run_id, rating, corrected_verdict, notes, reviewer, created_at
               FROM feedback ORDER BY created_at"""
        )
        rows = cur.fetchall()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(_json.dumps({
                "id": r[0], "run_id": r[1], "rating": r[2],
                "corrected_verdict": r[3], "notes": r[4],
                "reviewer": r[5], "created_at": r[6].isoformat() if r[6] else None,
            }) + "\n")
    typer.echo(f"wrote {len(rows)} feedback rows -> {out}")


if __name__ == "__main__":
    app()
