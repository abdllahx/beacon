import hashlib

import structlog

from beacon import db
from beacon.sources import RawArticle
from beacon.sources import gdelt as gdelt_src
from beacon.sources import newsapi as newsapi_src

log = structlog.get_logger()

DEFAULT_WILDFIRE_QUERY = '(wildfire OR "forest fire" OR "bush fire" OR "burn scar")'


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def upsert_article(conn, article: RawArticle) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO articles (source, url, url_hash, title, content, language, published_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url_hash) DO NOTHING
            RETURNING id
            """,
            (
                article.source,
                article.url,
                url_hash(article.url),
                article.title,
                article.content,
                article.language,
                article.published_at,
            ),
        )
        return cur.fetchone() is not None


def run_ingest(
    query: str = DEFAULT_WILDFIRE_QUERY,
    *,
    hours_back: int = 24,
    max_per_source: int = 50,
    sources: tuple[str, ...] = ("gdelt", "newsapi"),
) -> dict[str, int]:
    fetched: list[RawArticle] = []
    counts: dict[str, int] = {
        "gdelt_fetched": 0,
        "newsapi_fetched": 0,
        "inserted": 0,
        "duplicates": 0,
    }
    if "gdelt" in sources:
        gd = gdelt_src.fetch(query, max_records=max_per_source, timespan=f"{hours_back}h")
        counts["gdelt_fetched"] = len(gd)
        fetched.extend(gd)
    if "newsapi" in sources:
        na = newsapi_src.fetch(query, hours_back=hours_back, page_size=max_per_source)
        counts["newsapi_fetched"] = len(na)
        fetched.extend(na)
    with db.connect() as conn:
        for a in fetched:
            if upsert_article(conn, a):
                counts["inserted"] += 1
            else:
                counts["duplicates"] += 1
        conn.commit()
    log.info("ingest.done", **counts)
    return counts
