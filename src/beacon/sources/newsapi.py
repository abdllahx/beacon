from datetime import UTC, datetime, timedelta

import httpx
from dateutil import parser as dateparser

from beacon.config import get_settings
from beacon.sources.models import RawArticle

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch(
    query: str,
    *,
    hours_back: int = 24,
    page_size: int = 50,
    language: str = "en",
) -> list[RawArticle]:
    settings = get_settings()
    if not settings.newsapi_key:
        return []
    since = (datetime.now(UTC) - timedelta(hours=hours_back)).isoformat(timespec="seconds")
    params = {
        "q": query,
        "from": since,
        "sortBy": "publishedAt",
        "language": language,
        "pageSize": min(page_size, 100),
    }
    headers = {"X-Api-Key": settings.newsapi_key}
    with httpx.Client(timeout=30.0) as client:
        r = client.get(NEWSAPI_URL, params=params, headers=headers)
        if r.status_code >= 400:
            return []
        data = r.json()
    if data.get("status") != "ok":
        return []
    items = data.get("articles", []) or []
    out: list[RawArticle] = []
    for a in items:
        url = a.get("url")
        if not url:
            continue
        pub_raw = a.get("publishedAt")
        try:
            published_at = dateparser.parse(pub_raw) if pub_raw else None
        except (ValueError, TypeError):
            published_at = None
        content_parts = [a.get("description"), a.get("content")]
        content = "\n\n".join(p for p in content_parts if p) or None
        out.append(
            RawArticle(
                source="newsapi",
                url=url,
                title=a.get("title"),
                content=content,
                language=language,
                published_at=published_at,
            )
        )
    return out
