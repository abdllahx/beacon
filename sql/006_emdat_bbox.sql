ALTER TABLE emdat_events ADD COLUMN IF NOT EXISTS bbox GEOMETRY(Polygon, 4326);
ALTER TABLE emdat_events ADD COLUMN IF NOT EXISTS geocode_source TEXT;
CREATE INDEX IF NOT EXISTS emdat_bbox_gix ON emdat_events USING GIST (bbox);
