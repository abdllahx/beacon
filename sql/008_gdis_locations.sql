CREATE TABLE IF NOT EXISTS gdis_locations (
    id              BIGSERIAL PRIMARY KEY,
    gdis_id         INT,
    geo_id          INT,
    disasterno      TEXT NOT NULL,
    emdat_dis_no    TEXT,
    iso3            TEXT,
    country         TEXT,
    year            INT,
    disastertype    TEXT,
    level           INT,
    geolocation     TEXT,
    adm1            TEXT,
    adm2            TEXT,
    adm3            TEXT,
    location        TEXT,
    historical      INT,
    point           GEOMETRY(Point, 4326),
    raw             JSONB
);

CREATE INDEX IF NOT EXISTS gdis_disasterno_idx     ON gdis_locations (disasterno);
CREATE INDEX IF NOT EXISTS gdis_emdat_dis_no_idx   ON gdis_locations (emdat_dis_no);
CREATE INDEX IF NOT EXISTS gdis_point_gix          ON gdis_locations USING GIST (point);
CREATE INDEX IF NOT EXISTS gdis_year_type_idx      ON gdis_locations (year, disastertype);
