"""ACLED conflict-event loader (HF spec-relevant: ground truth for verification F1).

Authenticates with myACLED credentials via OAuth password grant, persists ONLY the
refresh_token (14d) to a gitignored file, refreshes access tokens (24h) on demand.
Username/password are never persisted to disk after the initial login.

Loads stratified samples by year + event_type. Default scope keeps to event types
that have plausible visual signature in 10m optical satellite imagery: Battles +
Explosions/Remote violence. Skips Protests / Riots / Strategic developments which
typically don't leave persistent damage.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog

from beacon import db
from beacon.config import get_settings

log = structlog.get_logger()

ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_API_URL = "https://acleddata.com/api/acled/read"
TOKEN_FILE = Path(".acled_tokens.json")  # gitignored

DEFAULT_EVENT_TYPES: tuple[str, ...] = ("Battles", "Explosions/Remote violence")


# --- token management ----------------------------------------------------


def login(username: str, password: str) -> dict:
    """One-time login. Persists tokens to .acled_tokens.json (gitignored)."""
    r = httpx.post(
        ACLED_TOKEN_URL,
        data={
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
            "scope": "authenticated",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    r.raise_for_status()
    tokens = r.json()
    if "access_token" not in tokens or "refresh_token" not in tokens:
        raise RuntimeError(f"Unexpected ACLED login response: {tokens}")
    tokens["_obtained_at"] = datetime.now(UTC).isoformat()
    TOKEN_FILE.write_text(json.dumps(tokens))
    log.info("acled.login_ok", username=username, expires_in=tokens.get("expires_in"))
    return tokens


def _refresh(refresh_token: str) -> dict:
    r = httpx.post(
        ACLED_TOKEN_URL,
        data={
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "client_id": "acled",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    r.raise_for_status()
    tokens = r.json()
    tokens["_obtained_at"] = datetime.now(UTC).isoformat()
    TOKEN_FILE.write_text(json.dumps(tokens))
    log.info("acled.token_refreshed")
    return tokens


def _get_access_token() -> str:
    if not TOKEN_FILE.exists():
        raise RuntimeError(
            "No .acled_tokens.json — run `beacon acled-login` first or set "
            "ACLED_USERNAME / ACLED_PASSWORD env vars."
        )
    tokens = json.loads(TOKEN_FILE.read_text())
    return tokens.get("access_token", "")


# --- API access ----------------------------------------------------------


def _fetch_with_auto_refresh(params: dict[str, Any]) -> dict:
    """Issue a GET against the ACLED API; on 401 refresh once and retry."""
    headers = {"Authorization": f"Bearer {_get_access_token()}"}
    r = httpx.get(ACLED_API_URL, params=params, headers=headers, timeout=60.0)
    if r.status_code == 401 and TOKEN_FILE.exists():
        refresh = json.loads(TOKEN_FILE.read_text()).get("refresh_token", "")
        if refresh:
            _refresh(refresh)
            headers = {"Authorization": f"Bearer {_get_access_token()}"}
            r = httpx.get(ACLED_API_URL, params=params, headers=headers, timeout=60.0)
    r.raise_for_status()
    return r.json()


def fetch_events(
    *,
    year: int,
    event_types: tuple[str, ...] = DEFAULT_EVENT_TYPES,
    country: str | None = None,
    page: int = 1,
    limit: int = 1000,
) -> list[dict]:
    """Fetch one page of ACLED events. ACLED API supports OR via :OR: separator."""
    params: dict[str, Any] = {
        "_format": "json",
        "year": year,
        "limit": limit,
        "page": page,
    }
    if event_types:
        # event_type=Battles:OR:event_type=Explosions/Remote violence
        params["event_type"] = ":OR:event_type=".join(event_types)
    if country:
        params["country"] = country
    payload = _fetch_with_auto_refresh(params)
    return payload.get("data", []) or []


# --- DB upsert ----------------------------------------------------------


def _safe_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _upsert_event(conn, e: dict) -> bool:
    """Insert one ACLED event. Returns True if new row was created."""
    event_id = e.get("event_id_cnty")
    if not event_id:
        return False
    lat = _safe_float(e.get("latitude"))
    lon = _safe_float(e.get("longitude"))
    point_wkt = f"POINT({lon} {lat})" if (lat is not None and lon is not None) else None
    event_date = e.get("event_date")
    try:
        ed = date.fromisoformat(event_date) if event_date else None
    except (ValueError, TypeError):
        ed = None
    if not ed:
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO acled_events
                (event_id_cnty, event_date, year, disorder_type, event_type,
                 sub_event_type, actor1, actor2, civilian_targeting,
                 iso, region, country, admin1, admin2, admin3, location,
                 point, geo_precision, source, source_scale, fatalities, notes, raw)
            VALUES
                (%s, %s, %s, %s, %s,
                 %s, %s, %s, %s,
                 %s, %s, %s, %s, %s, %s, %s,
                 CASE WHEN %s::text IS NOT NULL THEN ST_GeomFromText(%s, 4326) ELSE NULL END,
                 %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (event_id_cnty) DO NOTHING
            RETURNING id
            """,
            (
                event_id,
                ed,
                _safe_int(e.get("year")),
                e.get("disorder_type"),
                e.get("event_type"),
                e.get("sub_event_type"),
                e.get("actor1"),
                e.get("actor2"),
                e.get("civilian_targeting"),
                e.get("iso"),
                e.get("region"),
                e.get("country"),
                e.get("admin1"),
                e.get("admin2"),
                e.get("admin3"),
                e.get("location"),
                point_wkt,
                point_wkt,
                _safe_int(e.get("geo_precision")),
                e.get("source"),
                e.get("source_scale"),
                _safe_int(e.get("fatalities")),
                (e.get("notes") or "")[:5000],
                json.dumps(e),
            ),
        )
        return cur.fetchone() is not None


def load_country_years(
    countries: list[str],
    years: list[int],
    *,
    event_types: tuple[str, ...] = DEFAULT_EVENT_TYPES,
    page_size: int = 1000,
    max_pages_per_combo: int = 20,
) -> dict[str, int]:
    """Load events for the cross-product of (country × year). Paginates each combo."""
    counts = {"inserted": 0, "duplicates": 0, "errors": 0, "fetched": 0}
    with db.connect() as conn:
        for country in countries:
            for year in years:
                page = 1
                while page <= max_pages_per_combo:
                    try:
                        events = fetch_events(
                            year=year, event_types=event_types,
                            country=country, page=page, limit=page_size,
                        )
                    except Exception as e:
                        log.warning("acled.fetch_failed", country=country, year=year, page=page, err=str(e)[:160])
                        counts["errors"] += 1
                        break
                    if not events:
                        break
                    counts["fetched"] += len(events)
                    for ev in events:
                        try:
                            inserted = _upsert_event(conn, ev)
                            counts["inserted" if inserted else "duplicates"] += 1
                        except Exception:
                            log.exception("acled.upsert_error", event_id=ev.get("event_id_cnty"))
                            counts["errors"] += 1
                    conn.commit()
                    log.info("acled.page", country=country, year=year, page=page, n=len(events), **counts)
                    if len(events) < page_size:
                        break
                    page += 1
    return counts
