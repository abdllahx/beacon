import csv
import json
from datetime import UTC, date as date_type, datetime
from io import StringIO

import httpx
import structlog

from beacon import db
from beacon.config import get_settings

log = structlog.get_logger()

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_SOURCE = "VIIRS_SNPP_NRT"


def _maybe_float(x: str | None) -> float | None:
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None  # FIRMS confidence is 'l'/'n'/'h' for some products


def fetch_csv(
    *,
    area: tuple[float, float, float, float],
    days: int = 7,
    date: date_type | None = None,
    source: str = DEFAULT_SOURCE,
) -> str:
    settings = get_settings()
    if not settings.nasa_firms_key:
        raise RuntimeError("NASA_FIRMS_KEY missing — populate .env")
    coords = ",".join(str(round(c, 4)) for c in area)
    days = max(1, min(int(days), 5))  # FIRMS area endpoint caps at 5 days
    parts = [FIRMS_BASE, settings.nasa_firms_key, source, coords, str(days)]
    if date:
        parts.append(date.isoformat())
    url = "/".join(parts)
    with httpx.Client(timeout=60.0) as client:
        r = client.get(url)
    if r.status_code >= 400:
        log.warning("firms.error", status=r.status_code, body=r.text[:200])
        return ""
    return r.text


def parse_csv(csv_text: str, source: str) -> list[dict]:
    if not csv_text or not csv_text.strip():
        return []
    reader = csv.DictReader(StringIO(csv_text))
    out: list[dict] = []
    for row in reader:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (ValueError, KeyError):
            continue
        acq_date = row.get("acq_date", "").strip()
        acq_time = (row.get("acq_time") or "0000").strip().zfill(4)
        try:
            detected_at = datetime.strptime(
                f"{acq_date}T{acq_time}", "%Y-%m-%dT%H%M"
            ).replace(tzinfo=UTC)
        except ValueError:
            continue
        firms_id = f"{source}:{lat:.4f},{lon:.4f}@{acq_date}T{acq_time}"
        out.append(
            {
                "firms_id": firms_id,
                "lat": lat,
                "lon": lon,
                "detected_at": detected_at,
                "confidence": _maybe_float(row.get("confidence")),
                "frp": _maybe_float(row.get("frp")),
                "satellite": row.get("satellite"),
                "payload": dict(row),
            }
        )
    return out


def insert_events(conn, events: list[dict]) -> dict[str, int]:
    counts = {"inserted": 0, "duplicates": 0}
    with conn.cursor() as cur:
        for e in events:
            cur.execute(
                """
                INSERT INTO firms_events
                    (firms_id, detected_at, point, confidence, frp, satellite, payload)
                VALUES
                    (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s, %s::jsonb)
                ON CONFLICT (firms_id) DO NOTHING
                RETURNING id
                """,
                (
                    e["firms_id"],
                    e["detected_at"],
                    e["lon"],
                    e["lat"],
                    e.get("confidence"),
                    e.get("frp"),
                    e.get("satellite"),
                    json.dumps(e["payload"]),
                ),
            )
            if cur.fetchone() is not None:
                counts["inserted"] += 1
            else:
                counts["duplicates"] += 1
    return counts


def run_firms_load(
    *,
    area: tuple[float, float, float, float],
    days: int = 7,
    date: date_type | None = None,
    source: str = DEFAULT_SOURCE,
) -> dict[str, int]:
    raw = fetch_csv(area=area, days=days, date=date, source=source)
    events = parse_csv(raw, source=source)
    with db.connect() as conn:
        counts = insert_events(conn, events)
        conn.commit()
    counts["fetched"] = len(events)
    log.info("firms.load_done", source=source, days=days, **counts)
    return counts
