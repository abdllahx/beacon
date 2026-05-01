-- Collapse 10 separate per-modality columns on verification_runs into a single
-- imagery_metadata JSONB. Lets us add new modalities (NDVI, NDWI, Landsat) without
-- a schema migration each time, and gives readers (synth, Streamlit, eval_metrics)
-- a uniform shape to consume.
--
-- Final shape per row:
--   {
--     "s2":  {"before": {...}, "after": {...}},
--     "nbr": {"before": {...}, "after": {...}, "delta": {...}},
--     "s1":  {"before": {...}, "after": {...}, "change": {...}}
--   }

ALTER TABLE verification_runs
    ADD COLUMN IF NOT EXISTS imagery_metadata JSONB;

-- Backfill the JSONB from existing per-column data
UPDATE verification_runs
SET imagery_metadata = jsonb_strip_nulls(jsonb_build_object(
    's2', jsonb_strip_nulls(jsonb_build_object(
        'before', CASE WHEN before_tile_path IS NOT NULL
                       THEN jsonb_build_object('path', before_tile_path) END,
        'after',  CASE WHEN after_tile_path  IS NOT NULL
                       THEN jsonb_build_object('path', after_tile_path)  END
    )),
    'nbr', jsonb_strip_nulls(jsonb_build_object(
        'before', CASE WHEN nbr_before_path IS NOT NULL
                       THEN jsonb_build_object('path', nbr_before_path) END,
        'after',  CASE WHEN nbr_after_path  IS NOT NULL
                       THEN jsonb_build_object('path', nbr_after_path)  END,
        'delta',  CASE WHEN dnbr_path        IS NOT NULL
                       THEN jsonb_build_object(
                            'path', dnbr_path,
                            'burn_pct', dnbr_burn_pct) END
    )),
    's1', jsonb_strip_nulls(jsonb_build_object(
        'before', CASE WHEN s1_before_path IS NOT NULL
                       THEN jsonb_build_object('path', s1_before_path) END,
        'after',  CASE WHEN s1_after_path  IS NOT NULL
                       THEN jsonb_build_object('path', s1_after_path)  END,
        'change', CASE WHEN s1_change_path IS NOT NULL
                       THEN jsonb_build_object(
                            'path', s1_change_path,
                            'decrease_pct', s1_decrease_pct) END
    ))
))
WHERE imagery_metadata IS NULL;

-- Drop the deprecated per-modality columns. After this point persist_vision must
-- write to imagery_metadata, and readers must use the JSONB path expressions.
ALTER TABLE verification_runs
    DROP COLUMN IF EXISTS before_tile_path,
    DROP COLUMN IF EXISTS after_tile_path,
    DROP COLUMN IF EXISTS nbr_before_path,
    DROP COLUMN IF EXISTS nbr_after_path,
    DROP COLUMN IF EXISTS dnbr_path,
    DROP COLUMN IF EXISTS dnbr_burn_pct,
    DROP COLUMN IF EXISTS s1_before_path,
    DROP COLUMN IF EXISTS s1_after_path,
    DROP COLUMN IF EXISTS s1_change_path,
    DROP COLUMN IF EXISTS s1_decrease_pct;

CREATE INDEX IF NOT EXISTS verification_runs_imagery_gin
    ON verification_runs USING GIN (imagery_metadata);
