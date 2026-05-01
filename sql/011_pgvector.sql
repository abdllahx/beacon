-- Enable pgvector for sentence-similarity claim dedup and SigLIP-based VDR.
-- imresamu/postgis-pgvector image bundles both PostGIS and pgvector extensions.

CREATE EXTENSION IF NOT EXISTS vector;

-- Sentence-transformers all-MiniLM-L6-v2 emits 384-dim embeddings.
ALTER TABLE claims ADD COLUMN IF NOT EXISTS embedding vector(384);

-- HNSW index for fast cosine-similarity search across claims.
-- Used by the dedup node to find near-duplicate claims across sources.
CREATE INDEX IF NOT EXISTS claims_embedding_hnsw
    ON claims USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- SigLIP visual embeddings for VDR archive (will be wired in M2.B+C combined).
-- google/siglip-base-patch16-224 emits 768-dim embeddings.
CREATE TABLE IF NOT EXISTS tile_archive (
    id              BIGSERIAL PRIMARY KEY,
    tile_path       TEXT UNIQUE NOT NULL,
    bbox            GEOMETRY(Polygon, 4326),
    captured_at     DATE,
    disaster_type   TEXT,
    description     TEXT,
    embedding       vector(768)
);

CREATE INDEX IF NOT EXISTS tile_archive_embedding_hnsw
    ON tile_archive USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS tile_archive_bbox_gix
    ON tile_archive USING GIST (bbox);
