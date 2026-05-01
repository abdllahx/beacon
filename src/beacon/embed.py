"""Sentence-similarity embeddings for claim deduplication.

Uses sentence-transformers/all-MiniLM-L6-v2 (the canonical 384-dim short-text
embedder) running locally on CPU. Embeddings persist into `claims.embedding`
(pgvector vector(384)) with an HNSW cosine-similarity index for fast nearest-
neighbor lookup. Claim deduplication compares each claim against existing
claims via cosine distance — pairs above a similarity threshold are flagged as
near-duplicates that likely describe the same real-world event.

This is HF task #5 from the spec (Sentence Similarity).
"""

from __future__ import annotations

import structlog

from beacon import db

log = structlog.get_logger()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
DEFAULT_DEDUP_THRESHOLD = 0.85  # cosine similarity above which we flag as near-duplicate

_model_cache = None


def _model():
    """Load the embedding model once and cache. CPU is fine — ~80ms/text."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer

        log.info("embed.loading_model", model=EMBED_MODEL)
        _model_cache = SentenceTransformer(EMBED_MODEL)
    return _model_cache


def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a 384-dim list."""
    if not text or not text.strip():
        return [0.0] * EMBED_DIM
    vec = _model().encode(text[:2000], normalize_embeddings=True, convert_to_numpy=True)
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed many strings. Far faster than calling embed_text in a loop."""
    if not texts:
        return []
    safe = [t[:2000] if t else "" for t in texts]
    vecs = _model().encode(safe, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
    return vecs.tolist()


def _vector_literal(vec: list[float]) -> str:
    """Render a Python list as the pgvector text literal '[a, b, c]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def run_embed_backfill(*, limit: int | None = None, only_missing: bool = True) -> dict[str, int]:
    """Compute and persist embeddings for claims (missing ones by default)."""
    counts = {"embedded": 0, "skipped_empty": 0}
    sql = "SELECT id, raw_text FROM claims"
    where = []
    if only_missing:
        where.append("embedding IS NULL")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        if not rows:
            log.info("embed.backfill.no_rows")
            return counts
        # Batch in chunks to avoid huge memory spikes
        BATCH = 32
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            ids = [r[0] for r in batch]
            texts = [r[1] or "" for r in batch]
            non_empty_mask = [bool(t and t.strip()) for t in texts]
            if not any(non_empty_mask):
                counts["skipped_empty"] += len(batch)
                continue
            vecs = embed_texts(texts)
            with conn.cursor() as cur:
                for cid, vec, nonempty in zip(ids, vecs, non_empty_mask, strict=False):
                    if not nonempty:
                        counts["skipped_empty"] += 1
                        continue
                    cur.execute(
                        "UPDATE claims SET embedding = %s::vector WHERE id = %s",
                        (_vector_literal(vec), cid),
                    )
                    counts["embedded"] += 1
            conn.commit()
            log.info("embed.backfill.progress", done=counts["embedded"], total=len(rows))
    return counts


def find_near_duplicates(
    *,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    limit: int | None = None,
    same_event_type_only: bool = True,
) -> list[dict]:
    """Find pairs of claims with cosine similarity >= threshold.

    Uses pgvector's `<=>` cosine-distance operator. cosine_similarity = 1 - distance.
    Returns each unique pair (i,j with i<j) once, sorted by similarity descending.
    """
    type_filter = "AND a.event_type = b.event_type" if same_event_type_only else ""
    distance = 1 - threshold  # cosine distance threshold equivalent
    sql = f"""
        WITH pairs AS (
            SELECT
                a.id   AS claim_a,
                b.id   AS claim_b,
                a.embedding <=> b.embedding AS distance,
                a.event_type AS event_type,
                aa.title AS title_a,
                ab.title AS title_b
            FROM claims a
            JOIN claims b ON a.id < b.id
            JOIN articles aa ON aa.id = a.article_id
            JOIN articles ab ON ab.id = b.article_id
            WHERE a.embedding IS NOT NULL
              AND b.embedding IS NOT NULL
              AND a.embedding <=> b.embedding <= %s
              {type_filter}
        )
        SELECT claim_a, claim_b, distance, event_type, title_a, title_b
        FROM pairs
        ORDER BY distance ASC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (distance,))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return [
        {
            **dict(zip(cols, r, strict=False)),
            "similarity": round(1 - float(r[2]), 4),
        }
        for r in rows
    ]


def search_similar(query_text: str, *, k: int = 10) -> list[dict]:
    """Embed a query string and return the top-K most similar claims by cosine."""
    vec = embed_text(query_text)
    sql = """
        SELECT c.id, a.title, a.url, c.event_type,
               c.embedding <=> %s::vector AS distance
        FROM claims c
        JOIN articles a ON a.id = c.article_id
        WHERE c.embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT %s
    """
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (_vector_literal(vec), k))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return [
        {
            **dict(zip(cols, r, strict=False)),
            "similarity": round(1 - float(r[-1]), 4),
        }
        for r in rows
    ]
