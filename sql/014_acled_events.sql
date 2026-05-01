-- ACLED conflict events (battles, explosions, strategic developments).
-- Complements EM-DAT (natural disasters) for the spec's "EM-DAT/ACLED eval harness".
-- ACLED's event_id_cnty is a string like "GEO3998" — globally unique.
CREATE TABLE IF NOT EXISTS acled_events (
    id                  BIGSERIAL PRIMARY KEY,
    event_id_cnty       TEXT UNIQUE NOT NULL,
    event_date          DATE NOT NULL,
    year                INT,
    disorder_type       TEXT,
    event_type          TEXT,
    sub_event_type      TEXT,
    actor1              TEXT,
    actor2              TEXT,
    civilian_targeting  TEXT,
    iso                 TEXT,
    region              TEXT,
    country             TEXT,
    admin1              TEXT,
    admin2              TEXT,
    admin3              TEXT,
    location            TEXT,
    point               GEOMETRY(Point, 4326),
    geo_precision       INT,
    source              TEXT,
    source_scale        TEXT,
    fatalities          INT,
    notes               TEXT,
    raw                 JSONB
);

CREATE INDEX IF NOT EXISTS acled_event_date_idx  ON acled_events (event_date);
CREATE INDEX IF NOT EXISTS acled_country_idx     ON acled_events (country);
CREATE INDEX IF NOT EXISTS acled_event_type_idx  ON acled_events (event_type);
CREATE INDEX IF NOT EXISTS acled_fatalities_idx  ON acled_events (fatalities) WHERE fatalities IS NOT NULL;
CREATE INDEX IF NOT EXISTS acled_point_gix       ON acled_events USING GIST (point);
