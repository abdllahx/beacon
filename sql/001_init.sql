CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS articles (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    url_hash     CHAR(64) NOT NULL UNIQUE,
    title        TEXT,
    content      TEXT,
    language     TEXT,
    published_at TIMESTAMPTZ,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS articles_published_at_idx ON articles (published_at DESC);

CREATE TABLE IF NOT EXISTS claims (
    id            BIGSERIAL PRIMARY KEY,
    article_id    BIGINT NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    raw_text      TEXT NOT NULL,
    event_type    TEXT,
    locations     JSONB NOT NULL DEFAULT '[]'::jsonb,
    dates         JSONB NOT NULL DEFAULT '[]'::jsonb,
    actors        JSONB NOT NULL DEFAULT '[]'::jsonb,
    bbox          GEOMETRY(Polygon, 4326),
    admin_region  TEXT,
    geocode_score REAL,
    status        TEXT NOT NULL DEFAULT 'extracted',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS claims_bbox_gix ON claims USING GIST (bbox);
CREATE INDEX IF NOT EXISTS claims_status_idx ON claims (status);

CREATE TABLE IF NOT EXISTS verification_runs (
    id                  BIGSERIAL PRIMARY KEY,
    claim_id            BIGINT NOT NULL REFERENCES claims (id) ON DELETE CASCADE,
    status              TEXT NOT NULL DEFAULT 'pending',
    before_tile_path    TEXT,
    after_tile_path     TEXT,
    vision_verdict      JSONB,
    final_report_md     TEXT,
    final_verdict       JSONB,
    cost_usd            NUMERIC(10, 6),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS runs_claim_id_idx ON verification_runs (claim_id);

CREATE TABLE IF NOT EXISTS firms_events (
    id            BIGSERIAL PRIMARY KEY,
    firms_id      TEXT UNIQUE,
    detected_at   TIMESTAMPTZ NOT NULL,
    point         GEOMETRY(Point, 4326) NOT NULL,
    confidence    REAL,
    frp           REAL,
    satellite     TEXT,
    payload       JSONB
);

CREATE INDEX IF NOT EXISTS firms_point_gix ON firms_events USING GIST (point);
CREATE INDEX IF NOT EXISTS firms_detected_at_idx ON firms_events (detected_at DESC);
