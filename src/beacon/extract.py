import json

import structlog
from huggingface_hub import InferenceClient

from beacon import db
from beacon.config import get_settings

log = structlog.get_logger()

NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
ZSC_MODEL = "facebook/bart-large-mnli"

HAZARD_CATEGORIES: list[tuple[str, str | None, bool]] = [
    ("a wildfire or forest fire occurred at a specific location", "wildfire", True),
    ("a flood, severe rain, or river overflow occurred at a specific location", "flood", True),
    ("a tropical storm, hurricane, typhoon, cyclone, or severe wind event occurred at a specific location", "storm", True),
    ("an earthquake or seismic event occurred at a specific location", "earthquake", True),
    ("a landslide, mudslide, or debris flow occurred at a specific location", "landslide", True),
    ("a volcanic eruption occurred at a specific location", "volcanic", True),
    ("policy, government, or institutional news unrelated to a specific event", None, False),
    ("general news unrelated to natural hazards", None, False),
]
CANDIDATE_LABELS = [c[0] for c in HAZARD_CATEGORIES]
LABEL_TO_EVENT_TYPE = {c[0]: c[1] for c in HAZARD_CATEGORIES}
KEEP_LABELS = {c[0] for c in HAZARD_CATEGORIES if c[2]}

# Tunables imported from beacon.tunables — do not redefine here.
from beacon.tunables import RELEVANCE_THRESHOLD, MAX_INPUT_CHARS  # noqa: E402, F401


def _client() -> InferenceClient:
    settings = get_settings()
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN missing — populate .env")
    return InferenceClient(token=settings.hf_token)


def extract_entities(client: InferenceClient, text: str) -> list[dict]:
    out = client.token_classification(text[:MAX_INPUT_CHARS], model=NER_MODEL)
    return [
        {
            "text": e.word,
            "score": float(e.score),
            "start": e.start,
            "end": e.end,
            "entity_group": e.entity_group,
        }
        for e in out
    ]


def classify_relevance(client: InferenceClient, text: str) -> list[dict]:
    out = client.zero_shot_classification(
        text[:MAX_INPUT_CHARS],
        candidate_labels=CANDIDATE_LABELS,
        model=ZSC_MODEL,
    )
    return [{"label": e.label, "score": float(e.score)} for e in out]


def _bucket_entities(ents: list[dict]) -> tuple[list, list, list]:
    locations: list[dict] = []
    actors: list[dict] = []
    dates: list[dict] = []
    for e in ents:
        group = e.get("entity_group", "")
        item = {
            "text": e["text"],
            "score": e["score"],
            "start": e["start"],
            "end": e["end"],
        }
        if group == "LOC":
            locations.append(item)
        elif group in ("PER", "ORG"):
            actors.append(item)
        elif group == "DATE":
            dates.append(item)
    locations.sort(key=lambda x: -x["score"])
    actors.sort(key=lambda x: -x["score"])
    return locations, dates, actors


def _mark_status(conn, article_id: int, status: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE articles SET extract_status=%s, extract_attempted_at=now() WHERE id=%s",
            (status, article_id),
        )


def process_article(
    conn,
    client: InferenceClient,
    article_id: int,
    title: str | None,
    content: str | None,
) -> dict:
    text = ((title or "") + ("\n\n" + content if content else "")).strip()
    if not text:
        _mark_status(conn, article_id, "rejected")
        return {"status": "rejected", "reason": "empty"}
    try:
        scored = classify_relevance(client, text)
        top = scored[0] if scored else None
        top_label = top["label"] if top else None
        top_score = top["score"] if top else 0.0
        if top_label not in KEEP_LABELS or top_score < RELEVANCE_THRESHOLD:
            _mark_status(conn, article_id, "rejected")
            return {"status": "rejected", "top_score": round(top_score, 3)}
        ents = extract_entities(client, text)
        locations, dates, actors = _bucket_entities(ents)
        if not locations:
            _mark_status(conn, article_id, "rejected")
            return {"status": "rejected", "reason": "no_locations", "top_score": round(top_score, 3)}
        event_type = LABEL_TO_EVENT_TYPE.get(top_label, "unknown")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO claims (article_id, raw_text, event_type, locations, dates, actors)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                """,
                (
                    article_id,
                    text[:5000],
                    event_type,
                    json.dumps(locations),
                    json.dumps(dates),
                    json.dumps(actors),
                ),
            )
        _mark_status(conn, article_id, "kept")
        return {
            "status": "kept",
            "top_score": round(top_score, 3),
            "locations_n": len(locations),
            "actors_n": len(actors),
        }
    except Exception as e:
        log.exception("extract.error", article_id=article_id)
        _mark_status(conn, article_id, "error")
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)[:160]}"}


def run_extract(limit: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = {"kept": 0, "rejected": 0, "error": 0}
    client = _client()
    with db.connect() as conn:
        with conn.cursor() as cur:
            sql = (
                "SELECT id, title, content FROM articles "
                "WHERE extract_status IS NULL "
                "ORDER BY published_at DESC NULLS LAST"
            )
            if limit:
                sql += f" LIMIT {int(limit)}"
            cur.execute(sql)
            rows = cur.fetchall()
        for article_id, title, content in rows:
            res = process_article(conn, client, article_id, title, content)
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            log.info("extract.article", article_id=article_id, **res)
            conn.commit()
    return counts
