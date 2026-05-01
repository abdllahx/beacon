-- Cost log: every Claude / HF call writes a row. Source-of-truth for
-- per-verification cost analytics in the dashboard. Token counts are best-effort
-- (Agent SDK doesn't always surface them); when missing we estimate from char counts.
CREATE TABLE IF NOT EXISTS cost_events (
    id              BIGSERIAL PRIMARY KEY,
    run_id          BIGINT REFERENCES verification_runs (id) ON DELETE CASCADE,
    provider        TEXT NOT NULL,           -- 'claude' | 'huggingface' | 'planetary-computer'
    model           TEXT,                    -- e.g. 'claude-sonnet-4-5'
    operation       TEXT NOT NULL,           -- 'vision_vqa' | 'synthesize' | 'geocode_disambig' | ...
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        NUMERIC(10, 6),
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cost_events_run_idx     ON cost_events (run_id);
CREATE INDEX IF NOT EXISTS cost_events_op_idx      ON cost_events (operation);
CREATE INDEX IF NOT EXISTS cost_events_created_idx ON cost_events (created_at DESC);

-- Human-in-the-loop feedback: analyst reviews a verdict in the dashboard and
-- either confirms (thumbs up), corrects (thumbs down + corrected verdict),
-- or annotates (free-text). Routes back to a DSPy training set on next iteration.
CREATE TABLE IF NOT EXISTS feedback (
    id                BIGSERIAL PRIMARY KEY,
    run_id            BIGINT NOT NULL REFERENCES verification_runs (id) ON DELETE CASCADE,
    rating            TEXT NOT NULL CHECK (rating IN ('thumbs_up', 'thumbs_down', 'comment')),
    corrected_verdict TEXT CHECK (corrected_verdict IN ('supported', 'refuted', 'inconclusive')),
    notes             TEXT,
    reviewer          TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_run_idx    ON feedback (run_id);
CREATE INDEX IF NOT EXISTS feedback_rating_idx ON feedback (rating);
