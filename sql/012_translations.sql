-- Multi-language translation column for verification reports.
-- Shape: {"es": "...", "ar": "...", "fr": "..."}
ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS translations JSONB;
