#!/usr/bin/env bash
# Migrate local Beacon DB → Neon free tier (or Supabase). Pre-req: a Neon
# project with PostGIS + pgvector enabled (Settings → Extensions).
#
# Usage:
#   NEON_URL='postgresql://USER:PASS@HOST/DB?sslmode=require' ./scripts/migrate_to_neon.sh

set -euo pipefail

if [[ -z "${NEON_URL:-}" ]]; then
    echo "ERROR: set NEON_URL environment variable to the Neon connection string."
    exit 1
fi

DUMP_FILE="data/beacon_local_dump.sql"
mkdir -p data

echo "[1/4] dumping local Beacon DB → $DUMP_FILE"
docker exec beacon-db pg_dump -U beacon -d beacon \
    --no-owner --no-acl --quote-all-identifiers \
    > "$DUMP_FILE"

echo "[2/4] enabling extensions on Neon (idempotent)"
psql "$NEON_URL" -c "CREATE EXTENSION IF NOT EXISTS postgis; CREATE EXTENSION IF NOT EXISTS vector;"

echo "[3/4] restoring schema + data into Neon"
psql "$NEON_URL" -v ON_ERROR_STOP=1 -f "$DUMP_FILE"

echo "[4/4] verifying row counts"
psql "$NEON_URL" -c "
    SELECT 'verification_runs' AS table, COUNT(*) FROM verification_runs
    UNION ALL SELECT 'claims', COUNT(*) FROM claims
    UNION ALL SELECT 'benchmark_runs', COUNT(*) FROM benchmark_runs
    UNION ALL SELECT 'gdis_locations', COUNT(*) FROM gdis_locations;
"

echo "Done. Set DATABASE_URL=\$NEON_URL in .env and Streamlit Cloud secrets."
