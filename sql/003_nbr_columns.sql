ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS nbr_before_path TEXT;
ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS nbr_after_path TEXT;
ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS dnbr_path TEXT;
ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS dnbr_burn_pct REAL;
