"""Seed well-known historical wildfires as synthetic claims for end-to-end demo runs.

Each event is treated as if it were ingested through the normal pipeline (article + claim),
but with hand-curated bbox + admin_region so we don't depend on geocoder quality. Article
publish date is set to the event date so 'before/after' Sentinel-2 windows pull real imagery.
"""

import hashlib
import json
from datetime import UTC, datetime

import structlog

from beacon import db

log = structlog.get_logger()


DEMO_EVENTS = [
    {
        "key": "palisades-2025",
        "title": "Palisades Fire burns through Pacific Palisades, Los Angeles",
        "content": (
            "A wildfire broke out in Pacific Palisades, Los Angeles, on January 7, 2025, "
            "destroying thousands of structures and prompting widespread evacuations across the "
            "western Los Angeles foothills. The fire grew rapidly under Santa Ana wind conditions."
        ),
        "published_at": "2025-01-08T12:00:00+00:00",
        "bbox": (-118.60, 34.02, -118.50, 34.10),
        "admin_region": "Pacific Palisades, Los Angeles, California, USA",
        "primary_location": "Pacific Palisades",
    },
    {
        "key": "park-fire-2024",
        "title": "Park Fire scorches Butte and Tehama counties in Northern California",
        "content": (
            "The Park Fire ignited near Cohasset in Butte County, California, on July 24, 2024. "
            "It rapidly grew into one of the largest wildfires in California history, burning "
            "across more than 400,000 acres of forest and grassland."
        ),
        "published_at": "2024-07-26T12:00:00+00:00",
        "bbox": (-121.85, 39.85, -121.45, 40.15),
        "admin_region": "Butte and Tehama Counties, California, USA",
        "primary_location": "Cohasset",
    },
    {
        "key": "maui-2023",
        "title": "Lahaina wildfire devastates historic Maui town",
        "content": (
            "A wind-driven wildfire swept through the historic town of Lahaina on the west coast "
            "of Maui, Hawaii, on August 8, 2023. The fire destroyed much of the town center and "
            "displaced thousands of residents."
        ),
        "published_at": "2023-08-10T12:00:00+00:00",
        "bbox": (-156.70, 20.85, -156.65, 20.91),
        "admin_region": "Lahaina, Maui County, Hawaii, USA",
        "primary_location": "Lahaina",
    },
    {
        "key": "donnie-creek-2023",
        "title": "Donnie Creek fire becomes largest wildfire in BC history",
        "content": (
            "The Donnie Creek wildfire in northeastern British Columbia, ignited in May 2023, "
            "grew to over 600,000 hectares by August, becoming the largest single wildfire ever "
            "recorded in the province."
        ),
        "published_at": "2023-08-15T12:00:00+00:00",
        "bbox": (-123.10, 56.90, -122.50, 57.30),
        "admin_region": "Northeastern British Columbia, Canada",
        "primary_location": "Donnie Creek",
    },
    {
        "key": "rhodes-2023",
        "title": "Wildfires force mass evacuations on Greek island of Rhodes",
        "content": (
            "Wildfires burning across the Greek island of Rhodes in late July 2023 forced the "
            "evacuation of over 19,000 tourists and residents — the largest evacuation in Greek "
            "history. Mediterranean scrubland and forested hills near Lardos and Kiotari were "
            "heavily impacted."
        ),
        "published_at": "2023-07-25T12:00:00+00:00",
        "bbox": (28.00, 36.04, 28.20, 36.18),
        "admin_region": "Rhodes, South Aegean, Greece",
        "primary_location": "Rhodes",
    },
]


def _bbox_wkt(bbox: tuple[float, float, float, float]) -> str:
    w, s, e, n = bbox
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def seed_event(conn, event: dict) -> int:
    """Insert a synthetic article + claim. Returns claim_id. Idempotent on URL hash."""
    url = f"demo://{event['key']}"
    url_h = hashlib.sha256(url.encode()).hexdigest()
    raw_text = (event["title"] or "") + "\n\n" + (event["content"] or "")
    locations = json.dumps([{"text": event["primary_location"], "score": 0.99}])

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO articles (source, url, url_hash, title, content, language, published_at, extract_status)
            VALUES ('demo', %s, %s, %s, %s, 'en', %s, 'kept')
            ON CONFLICT (url_hash) DO UPDATE SET extract_status='kept'
            RETURNING id
            """,
            (url, url_h, event["title"], event["content"], event["published_at"]),
        )
        article_id = cur.fetchone()[0]

        cur.execute("SELECT id FROM claims WHERE article_id=%s", (article_id,))
        existing = cur.fetchone()
        if existing:
            return existing[0]

        cur.execute(
            """
            INSERT INTO claims (article_id, raw_text, event_type, locations,
                                bbox, admin_region, geocode_score, status)
            VALUES (%s, %s, 'wildfire', %s::jsonb,
                    ST_GeomFromText(%s, 4326), %s, 0.9, 'geocoded')
            RETURNING id
            """,
            (article_id, raw_text, locations, _bbox_wkt(event["bbox"]), event["admin_region"]),
        )
        return cur.fetchone()[0]


def seed_all() -> list[int]:
    ids: list[int] = []
    with db.connect() as conn:
        for event in DEMO_EVENTS:
            cid = seed_event(conn, event)
            ids.append(cid)
            log.info("demo_seed.event", key=event["key"], claim_id=cid)
        conn.commit()
    return ids


__all__ = ["DEMO_EVENTS", "seed_all", "seed_event"]
