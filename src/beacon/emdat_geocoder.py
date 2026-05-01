"""Supplemental geocoder for EM-DAT events.

Most non-earthquake EM-DAT records lack native Lat/Lon (Wildfire 0%, Storm 0.3%,
Flood 7.5%) but ship a structured `Admin Units` JSON field with adm0/1/2 names.
This module turns those names into bounding boxes via Nominatim so we can include
floods, storms, and wildfires in the 500-event benchmark.

Two passes:
1. `populate_native_bbox()` — for events with a Latitude/Longitude point, set a
   small (~10 km) square bbox and tag geocode_source='native'.
2. `run_admin_geocoder()` — for events with NULL point + Admin Units JSON, hit
   Nominatim once per event (1.1s throttle), persist point + bbox, tag
   geocode_source='admin_nominatim'.
"""

import json
import re
from typing import Any

import structlog

from beacon import db
from beacon.geocode import _bbox_wkt, nominatim_search

_PARENS_RE = re.compile(r"\([^)]*\)")
_SPLIT_RE = re.compile(r"[,;]")
_DROP_WORDS = {
    "province", "provinces", "region", "regions", "district", "districts",
    "community", "communities", "area", "areas", "city", "cities",
    "island", "islands", "isl.", "and",
}

log = structlog.get_logger()


def populate_native_bbox() -> int:
    """Generate ~10 km bbox squares around events that have native lat/lon points."""
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE emdat_events
            SET bbox = ST_GeomFromText(
                'POLYGON((' ||
                  (ST_X(point) - 0.05) || ' ' || (ST_Y(point) - 0.05) || ', ' ||
                  (ST_X(point) + 0.05) || ' ' || (ST_Y(point) - 0.05) || ', ' ||
                  (ST_X(point) + 0.05) || ' ' || (ST_Y(point) + 0.05) || ', ' ||
                  (ST_X(point) - 0.05) || ' ' || (ST_Y(point) + 0.05) || ', ' ||
                  (ST_X(point) - 0.05) || ' ' || (ST_Y(point) - 0.05) ||
                '))', 4326),
                geocode_source = 'native'
            WHERE point IS NOT NULL AND bbox IS NULL
            """
        )
        n = cur.rowcount
        conn.commit()
    log.info("emdat.populate_native_bbox", updated=n)
    return n


def _query_from_admin_units(admin_units_json: Any, country: str | None) -> str | None:
    if not admin_units_json:
        return None
    if isinstance(admin_units_json, str):
        try:
            adm = json.loads(admin_units_json)
        except (ValueError, TypeError):
            return None
    else:
        adm = admin_units_json
    if not isinstance(adm, list) or not adm:
        return None
    first = adm[0]
    if not isinstance(first, dict):
        return None
    for key in ("adm2_name", "adm1_name", "adm0_name"):
        v = first.get(key)
        if v:
            base = str(v).strip()
            if country and country.lower() not in base.lower():
                return f"{base}, {country}"
            return base
    return None


def _split_location_candidates(location: str) -> list[str]:
    """Pull individual place names out of free-text location like
    'Whakatāne, Ōhope, and Thornton communities (Bay of Plenty)'.
    """
    if not location:
        return []
    s = _PARENS_RE.sub("", location)
    raw_parts = _SPLIT_RE.split(s)
    out: list[str] = []
    for part in raw_parts:
        # Drop generic suffixes
        words = [w for w in part.split() if w.lower() not in _DROP_WORDS]
        cleaned = " ".join(words).strip(" .'\"")
        if len(cleaned) >= 3 and cleaned.lower() != "and":
            out.append(cleaned)
    return out


def _query_from_location(location: str | None, country: str | None) -> str | None:
    """Pick the first specific place name and combine with country."""
    candidates = _split_location_candidates(location or "")
    if not candidates:
        return None
    base = candidates[0]
    if country and country.lower() not in base.lower():
        return f"{base}, {country}"
    return base


def _build_query(
    admin_units_json: Any, country: str | None, location: str | None
) -> tuple[str, str] | None:
    """Return (query, source_tag) from the most specific signal available."""
    q = _query_from_admin_units(admin_units_json, country)
    if q:
        return (q, "admin_nominatim")
    q = _query_from_location(location, country)
    if q:
        return (q, "location_nominatim")
    if country:
        return (country, "country_nominatim")
    return None


def run_admin_geocoder(
    limit: int | None = None,
    *,
    only_event_types: tuple[str, ...] | None = None,
) -> dict[str, int]:
    counts = {
        "geocoded_admin": 0,
        "geocoded_location": 0,
        "geocoded_country": 0,
        "no_signal": 0,
        "nominatim_empty": 0,
        "no_bbox_in_result": 0,
        "errors": 0,
    }
    sql = """
        SELECT id, admin_units, country, location, disaster_type
        FROM emdat_events
        WHERE point IS NULL
    """
    args: list[Any] = []
    if only_event_types:
        sql += " AND disaster_type = ANY(%s)"
        args.append(list(only_event_types))
    sql += " ORDER BY start_date DESC NULLS LAST"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        log.info("emdat.geocode.start", n=len(rows))
        for event_id, admin_units, country, location, _disaster_type in rows:
            built = _build_query(admin_units, country, location)
            if not built:
                counts["no_signal"] += 1
                continue
            query, source_tag = built
            try:
                results = nominatim_search(query, limit=3)
            except Exception as e:
                log.warning("emdat.geocode.error", id=event_id, error=str(e)[:160])
                counts["errors"] += 1
                continue
            if not results:
                counts["nominatim_empty"] += 1
                continue
            top = max(results, key=lambda r: float(r.get("importance", 0)))
            bb = top.get("boundingbox")
            if not bb or len(bb) != 4:
                counts["no_bbox_in_result"] += 1
                continue
            try:
                lat_min, lat_max, lon_min, lon_max = map(float, bb)
            except (ValueError, TypeError):
                counts["no_bbox_in_result"] += 1
                continue
            cx, cy = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
            wkt = _bbox_wkt(lat_min, lat_max, lon_min, lon_max)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE emdat_events
                    SET point = ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                        bbox = ST_GeomFromText(%s, 4326),
                        geocode_source = %s
                    WHERE id = %s
                    """,
                    (cx, cy, wkt, source_tag, event_id),
                )
            conn.commit()
            tag_key = {
                "admin_nominatim": "geocoded_admin",
                "location_nominatim": "geocoded_location",
                "country_nominatim": "geocoded_country",
            }[source_tag]
            counts[tag_key] += 1
            n_done = counts["geocoded_admin"] + counts["geocoded_location"] + counts["geocoded_country"]
            if n_done % 25 == 0:
                log.info("emdat.geocode.progress", n=n_done, **counts)
    log.info("emdat.geocode.done", **counts)
    return counts
