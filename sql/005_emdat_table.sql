CREATE TABLE IF NOT EXISTS emdat_events (
    id                          BIGSERIAL PRIMARY KEY,
    dis_no                      TEXT UNIQUE NOT NULL,
    disaster_type               TEXT,
    disaster_subtype            TEXT,
    country                     TEXT,
    iso                          TEXT,
    location                    TEXT,
    point                       GEOMETRY(Point, 4326),
    start_date                  DATE,
    end_date                    DATE,
    magnitude                   REAL,
    magnitude_scale             TEXT,
    total_deaths                INT,
    total_affected              INT,
    total_damage_kusd           REAL,
    admin_units                 JSONB,
    raw                         JSONB
);

CREATE INDEX IF NOT EXISTS emdat_point_gix ON emdat_events USING GIST (point);
CREATE INDEX IF NOT EXISTS emdat_start_date_idx ON emdat_events (start_date);
CREATE INDEX IF NOT EXISTS emdat_type_idx ON emdat_events (disaster_type);
CREATE INDEX IF NOT EXISTS emdat_country_idx ON emdat_events (country);
