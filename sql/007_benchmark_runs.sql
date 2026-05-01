CREATE TABLE IF NOT EXISTS benchmark_runs (
    id                  BIGSERIAL PRIMARY KEY,
    emdat_event_id      BIGINT NOT NULL REFERENCES emdat_events (id) ON DELETE CASCADE,
    claim_id            BIGINT REFERENCES claims (id) ON DELETE SET NULL,
    beacon_run_id       BIGINT REFERENCES verification_runs (id) ON DELETE SET NULL,
    expected_verdict    TEXT NOT NULL DEFAULT 'supported',
    beacon_verdict      TEXT,
    beacon_confidence   REAL,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS benchmark_emdat_uniq ON benchmark_runs (emdat_event_id);
CREATE INDEX IF NOT EXISTS benchmark_run_idx ON benchmark_runs (beacon_run_id);
