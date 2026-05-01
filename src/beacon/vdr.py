"""Visual Document Retrieval — embed satellite tiles via SigLIP, store in pgvector
`tile_archive` table, retrieve top-K most similar archived tiles for a query image.

HF task: Visual Document Retrieval. Same SigLIP model that powers tile classification
also powers retrieval — single model, two tasks (Zero-Shot Image Classification
+ Visual Document Retrieval) per the spec.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import structlog

from beacon import db
from beacon.siglip import embed_image, embed_text

log = structlog.get_logger()


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def add_tile(
    tile_path: str,
    *,
    bbox_wkt: str | None = None,
    captured_at: date | None = None,
    disaster_type: str | None = None,
    description: str | None = None,
) -> int:
    """Embed a tile + insert into tile_archive. Returns archive id (re-embeds on conflict)."""
    embedding = embed_image(tile_path)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tile_archive (tile_path, bbox, captured_at, disaster_type, description, embedding)
            VALUES (
                %s,
                CASE WHEN %s::text IS NOT NULL THEN ST_GeomFromText(%s, 4326) ELSE NULL END,
                %s, %s, %s, %s::vector
            )
            ON CONFLICT (tile_path) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    description = COALESCE(EXCLUDED.description, tile_archive.description)
            RETURNING id
            """,
            (tile_path, bbox_wkt, bbox_wkt, captured_at, disaster_type, description, _vector_literal(embedding)),
        )
        rid = cur.fetchone()[0]
        conn.commit()
    return rid


def search_archive(query_image_path: str, *, k: int = 5) -> list[dict]:
    """Return top-K most similar tiles in tile_archive for the query image."""
    embedding = embed_image(query_image_path)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, tile_path, captured_at, disaster_type, description,
                   embedding <=> %s::vector AS distance
            FROM tile_archive
            WHERE embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT %s
            """,
            (_vector_literal(embedding), k),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return [
        {
            **dict(zip(cols, r, strict=False)),
            "similarity": round(1 - float(r[-1]), 4),
        }
        for r in rows
    ]


def search_by_text(query: str, *, k: int = 5) -> list[dict]:
    """Cross-modal retrieval: embed a text query (e.g. 'severe burn scar in foothills')
    and find archived tiles that match. SigLIP aligns image + text in the same space."""
    embedding = embed_text(query)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, tile_path, captured_at, disaster_type, description,
                   embedding <=> %s::vector AS distance
            FROM tile_archive
            WHERE embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT %s
            """,
            (_vector_literal(embedding), k),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return [
        {
            **dict(zip(cols, r, strict=False)),
            "similarity": round(1 - float(r[-1]), 4),
        }
        for r in rows
    ]


def seed_demo_archive() -> dict[str, int]:
    """Seed the tile archive with the existing demo wildfire AFTER tiles.
    Establishes a small but real corpus so VDR has something to retrieve."""
    counts = {"added": 0, "skipped": 0}
    demo_specs = [
        ("data/tiles/claim_11_after.png", "wildfire", "Pacific Palisades, January 2025 — large burn scar"),
        ("data/tiles/claim_12_after.png", "wildfire", "Park Fire, July 2024 — Butte/Tehama foothills burn"),
        ("data/tiles/claim_13_after.png", "wildfire", "Lahaina Maui, August 2023 — coastal urban burn"),
        ("data/tiles/claim_14_after.png", "wildfire", "Donnie Creek BC, 2023 — boreal forest burn"),
        ("data/tiles/claim_15_after.png", "wildfire", "Rhodes Greece, July 2023 — Mediterranean scrub burn"),
    ]
    for path, dtype, desc in demo_specs:
        if not Path(path).exists():
            counts["skipped"] += 1
            continue
        try:
            add_tile(path, disaster_type=dtype, description=desc)
            counts["added"] += 1
        except Exception as e:
            log.warning("vdr.seed_failed", path=path, error=str(e)[:120])
            counts["skipped"] += 1
    log.info("vdr.seed_done", **counts)
    return counts
