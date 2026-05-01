"""Load the GDIS dataset (Rosvold & Buhaug 2021) into Postgres.

GDIS is a peer-reviewed extension of EM-DAT with subnational geocoded disaster
locations from 1960-2018. We use it as ground truth for the verification benchmark
on events that overlap with the Sentinel-2 era (2017-2018, ~1k rows).

Joins to emdat_events via (GDIS.disasterno + '-' + iso3) = emdat_events.dis_no.
"""

import json
import math
from pathlib import Path

import pandas as pd
import structlog

from beacon import db

log = structlog.get_logger()


def _safe_int(v) -> int | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def load_gdis(csv_path: Path) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    log.info("gdis.read", rows=len(df), cols=len(df.columns))
    counts = {"inserted": 0, "skipped_no_coords": 0, "skipped_no_disasterno": 0, "errors": 0}
    with db.connect() as conn, conn.cursor() as cur:
        for _, row in df.iterrows():
            try:
                lat = _safe_float(row.get("latitude"))
                lon = _safe_float(row.get("longitude"))
                if lat is None or lon is None:
                    counts["skipped_no_coords"] += 1
                    continue
                disasterno = _safe_str(row.get("disasterno"))
                if not disasterno:
                    counts["skipped_no_disasterno"] += 1
                    continue
                iso3 = _safe_str(row.get("iso3"))
                emdat_dis_no = f"{disasterno}-{iso3}" if iso3 else disasterno
                cur.execute(
                    """
                    INSERT INTO gdis_locations
                        (gdis_id, geo_id, disasterno, emdat_dis_no, iso3, country,
                         year, disastertype, level, geolocation, adm1, adm2, adm3,
                         location, historical, point, raw)
                    VALUES
                        (%s, %s, %s, %s, %s, %s,
                         %s, %s, %s, %s, %s, %s, %s,
                         %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s::jsonb)
                    """,
                    (
                        _safe_int(row.get("id")),
                        _safe_int(row.get("geo_id")),
                        disasterno,
                        emdat_dis_no,
                        iso3,
                        _safe_str(row.get("country")),
                        _safe_int(row.get("year")),
                        _safe_str(row.get("disastertype")),
                        _safe_int(row.get("level")),
                        _safe_str(row.get("geolocation")),
                        _safe_str(row.get("adm1")),
                        _safe_str(row.get("adm2")),
                        _safe_str(row.get("adm3")),
                        _safe_str(row.get("location")),
                        _safe_int(row.get("historical")),
                        lon,
                        lat,
                        json.dumps(row.dropna().to_dict(), default=str),
                    ),
                )
                counts["inserted"] += 1
            except Exception as e:
                log.warning("gdis.row_error", error=str(e)[:160])
                counts["errors"] += 1
                conn.rollback()
                continue
        conn.commit()
    log.info("gdis.load_done", **counts)
    return counts
