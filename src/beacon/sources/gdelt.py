import time
from datetime import UTC, datetime

import httpx
import structlog

from beacon.sources.models import RawArticle

log = structlog.get_logger()

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "beacon/0.0.1 (geospatial-event-verification; portfolio project)"


def _parse_seendate(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def fetch(
    query: str,
    *,
    max_records: int = 50,
    timespan: str = "24h",
    source_lang: str | None = "english",
) -> list[RawArticle]:
    """Query GDELT 2.0 DOC API for articles matching a keyword query.

    GDELT does not require an API key. Only headlines and metadata are returned —
    no article bodies. timespan format: "{N}h" or "{N}d".
    """
    full_query = f"{query} sourcelang:{source_lang}" if source_lang else query
    params = {
        "query": full_query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "timespan": timespan,
        "sort": "datedesc",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    with httpx.Client(timeout=30.0, headers=headers) as client:
        for attempt in range(3):
            r = client.get(GDELT_DOC_URL, params=params)
            if r.status_code == 429:
                log.warning("gdelt.rate_limited", attempt=attempt + 1)
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                log.warning("gdelt.http_error", status=r.status_code, body=r.text[:200])
                return []
            break
        else:
            log.warning("gdelt.rate_limited_giving_up")
            return []
        if not r.text.strip():
            return []
        try:
            data = r.json()
        except ValueError:
            log.warning("gdelt.invalid_json", body=r.text[:200])
            return []
    items = data.get("articles", []) or []
    out: list[RawArticle] = []
    for a in items:
        url = a.get("url")
        if not url:
            continue
        out.append(
            RawArticle(
                source="gdelt",
                url=url,
                title=a.get("title"),
                language=a.get("language"),
                published_at=_parse_seendate(a.get("seendate")),
            )
        )
    return out
