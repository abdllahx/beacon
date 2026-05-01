ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS s1_before_path TEXT;
ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS s1_after_path TEXT;
ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS s1_change_path TEXT;
ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS s1_decrease_pct REAL;
