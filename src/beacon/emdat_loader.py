"""Load EM-DAT public Excel exports into emdat_events.

Run from CLI: `beacon emdat-load --file data/raw/emdat.xlsx`
"""

import json
import math
from datetime import date
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


def _build_date(year, month, day) -> date | None:
    y = _safe_int(year)
    if not y:
        return None
    m = _safe_int(month) or 1
    d = _safe_int(day) or 1
    m = max(1, min(12, m))
    d = max(1, min(28, d)) if m == 2 else max(1, min(31, d))
    try:
        return date(y, m, d)
    except (ValueError, TypeError):
        return None


def _parse_admin_units(v) -> str | None:
    """Admin Units cells arrive as JSON-formatted strings; return canonical JSON or None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        try:
            return json.dumps(json.loads(v))
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(v, list | dict):
        return json.dumps(v)
    return None


def load_emdat(xlsx_path: Path) -> dict[str, int]:
    df = pd.read_excel(xlsx_path)
    log.info("emdat.read", rows=len(df), cols=len(df.columns))

    counts = {"inserted": 0, "duplicates": 0, "skipped": 0}
    with db.connect() as conn, conn.cursor() as cur:
        for _, row in df.iterrows():
            dis_no = _safe_str(row.get("DisNo."))
            if not dis_no:
                counts["skipped"] += 1
                continue
            lat = _safe_float(row.get("Latitude"))
            lon = _safe_float(row.get("Longitude"))
            point_wkt = f"POINT({lon} {lat})" if lat is not None and lon is not None else None
            start_dt = _build_date(row.get("Start Year"), row.get("Start Month"), row.get("Start Day"))
            end_dt = _build_date(row.get("End Year"), row.get("End Month"), row.get("End Day"))
            admin_units = _parse_admin_units(row.get("Admin Units"))
            try:
                cur.execute(
                    """
                    INSERT INTO emdat_events
                        (dis_no, disaster_type, disaster_subtype, country, iso, location,
                         point, start_date, end_date,
                         magnitude, magnitude_scale,
                         total_deaths, total_affected, total_damage_kusd, admin_units)
                    VALUES (%s, %s, %s, %s, %s, %s,
                            CASE WHEN %s::text IS NOT NULL THEN ST_GeomFromText(%s, 4326) ELSE NULL END,
                            %s, %s,
                            %s, %s,
                            %s, %s, %s, %s::jsonb)
                    ON CONFLICT (dis_no) DO NOTHING
                    RETURNING id
                    """,
                    (
                        dis_no,
                        _safe_str(row.get("Disaster Type")),
                        _safe_str(row.get("Disaster Subtype")),
                        _safe_str(row.get("Country")),
                        _safe_str(row.get("ISO")),
                        _safe_str(row.get("Location")),
                        point_wkt,
                        point_wkt,
                        start_dt,
                        end_dt,
                        _safe_float(row.get("Magnitude")),
                        _safe_str(row.get("Magnitude Scale")),
                        _safe_int(row.get("Total Deaths")),
                        _safe_int(row.get("Total Affected")),
                        _safe_float(row.get("Total Damage ('000 US$)")),
                        admin_units,
                    ),
                )
                if cur.fetchone() is not None:
                    counts["inserted"] += 1
                else:
                    counts["duplicates"] += 1
            except Exception as e:
                log.warning("emdat.row_error", dis_no=dis_no, error=str(e)[:160])
                counts["skipped"] += 1
                conn.rollback()
                continue
        conn.commit()
    log.info("emdat.load_done", **counts)
    return counts
