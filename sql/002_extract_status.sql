ALTER TABLE articles ADD COLUMN IF NOT EXISTS extract_status TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS extract_attempted_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS articles_extract_status_idx ON articles (extract_status);
