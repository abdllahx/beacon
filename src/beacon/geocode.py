import time

import httpx
import structlog

from beacon import claude, db

log = structlog.get_logger()

from beacon.tunables import (
    MAX_NER_CANDIDATES,
    MIN_NER_SCORE,
    NOMINATIM_REQUEST_INTERVAL_S as REQUEST_INTERVAL_S,
    NOMINATIM_RESULTS_PER_LOC,
)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "beacon/0.0.1 (geospatial-event-verification; portfolio project)"

_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.time() - _last_request_at
    if elapsed < REQUEST_INTERVAL_S:
        time.sleep(REQUEST_INTERVAL_S - elapsed)
    _last_request_at = time.time()


class NominatimError(RuntimeError):
    """Raised when a Nominatim call fails in a way the caller should know about."""

    def __init__(self, kind: str, *, status: int | None = None, query: str = "", detail: str = ""):
        self.kind = kind
        self.status = status
        self.query = query
        self.detail = detail
        super().__init__(f"nominatim {kind} (status={status}, query={query!r}): {detail[:160]}")


def nominatim_search(query: str, *, limit: int = 5) -> list[dict]:
    """Search Nominatim. Distinguishes failure modes so the caller can handle them:

    - timeout / connection errors → NominatimError("network", ...)
    - 429 rate-limited → NominatimError("rate_limited", status=429, ...)
    - 5xx server error → NominatimError("server_error", status=5xx, ...)
    - 4xx (other) bad request → NominatimError("bad_request", status=4xx, ...)
    - parse error → NominatimError("invalid_response", ...)
    - empty result → returns [] (NOT an error — geocoder may legitimately have nothing)
    """
    _throttle()
    headers = {"User-Agent": USER_AGENT}
    params = {"q": query, "format": "jsonv2", "limit": limit, "addressdetails": 1}
    try:
        with httpx.Client(timeout=30.0, headers=headers) as client:
            r = client.get(NOMINATIM_URL, params=params)
    except httpx.TimeoutException as e:
        log.warning("nominatim.timeout", query=query, detail=str(e)[:160])
        raise NominatimError("timeout", query=query, detail=str(e)) from e
    except httpx.RequestError as e:
        log.warning("nominatim.network_error", query=query, detail=str(e)[:160])
        raise NominatimError("network", query=query, detail=str(e)) from e
    if r.status_code == 429:
        log.warning("nominatim.rate_limited", query=query)
        raise NominatimError("rate_limited", status=429, query=query, detail=r.text[:200])
    if 500 <= r.status_code < 600:
        log.warning("nominatim.server_error", status=r.status_code, query=query)
        raise NominatimError("server_error", status=r.status_code, query=query, detail=r.text[:200])
    if 400 <= r.status_code < 500:
        log.warning("nominatim.bad_request", status=r.status_code, query=query, body=r.text[:200])
        raise NominatimError("bad_request", status=r.status_code, query=query, detail=r.text[:200])
    try:
        return r.json()
    except ValueError as e:
        log.warning("nominatim.invalid_response", query=query, detail=str(e)[:160])
        raise NominatimError("invalid_response", query=query, detail=str(e)) from e


def _bbox_wkt(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
    return (
        f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, "
        f"{lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"
    )


def _bbox_from_nominatim(item: dict) -> str | None:
    bb = item.get("boundingbox")
    if not bb or len(bb) != 4:
        return None
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, bb)
    except (ValueError, TypeError):
        return None
    return _bbox_wkt(lat_min, lat_max, lon_min, lon_max)


def _viable_locations(locations: list[dict], *, max_n: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for loc in locations or []:
        text = (loc.get("text") or "").strip()
        if (
            float(loc.get("score", 0)) >= MIN_NER_SCORE
            and len(text) >= 3
            and text.lower() not in seen
        ):
            out.append(text)
            seen.add(text.lower())
        if len(out) >= max_n:
            break
    return out


def _claude_pick_most_specific(
    article_text: str,
    options: list[dict],
) -> int | None:
    """One-shot Claude disambiguation across ALL (NER location, OSM match) candidates.

    Each option dict has keys: loc_text, display_name, type, importance. Returns the
    index in `options` of Claude's pick, or None if Claude rejects all. Crucially,
    the prompt instructs Claude to *prefer specific places* (city/county) over
    countries/states — solving the BERT-NER frequency bias toward broad regions.
    """
    if not options:
        return None
    if len(options) == 1:
        return 0
    listing = "\n".join(
        f"  {i + 1}. \"{o['loc_text']}\" → {o['display_name']} (osm_type={o.get('type')}, importance={o.get('importance', 0):.3f})"
        for i, o in enumerate(options)
    )
    prompt = f"""You are picking the most precise event location from a set of candidates.

Article excerpt:
{article_text[:1500]}

A NER pass over the article extracted multiple location strings. Each was queried
against OpenStreetMap. The (extracted_string → OSM_match) pairs are below:

{listing}

Pick the SINGLE candidate that BEST matches the actual location of the event described
in the article. Strongly prefer SPECIFIC places — a city/county/district is better than
a state/province, and a state/province is better than a country. Events happen at
places, not across whole nations.

Reply with ONLY a number 1-{len(options)} or NONE if no candidate plausibly matches."""
    try:
        raw = claude.ask(prompt)
    except Exception as e:
        # Don't silently default to picking option 0 — that's how bad geocodes get
        # baked into benchmark numbers. Surface the failure to the caller.
        log.warning("claude.disambiguate_failed", error=str(e)[:160], n_options=len(options))
        raise ClaudeDisambiguationError(reason="exception", detail=str(e)) from e
    token = (raw or "").strip().split("\n")[0].split()[0] if raw else ""
    if token.upper().startswith("NONE"):
        return None
    try:
        idx = int(token) - 1
        if 0 <= idx < len(options):
            return idx
    except ValueError:
        pass
    log.warning("claude.disambiguate_unparseable", raw=(raw or "")[:120])
    raise ClaudeDisambiguationError(reason="unparseable", detail=(raw or "")[:200])


class ClaudeDisambiguationError(RuntimeError):
    """Raised when Claude's location-pick output can't be used (exception or unparseable)."""

    def __init__(self, *, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"claude_disambiguate_{reason}: {detail[:160]}")


def _set_status(conn, claim_id: int, status: str) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE claims SET status=%s WHERE id=%s", (status, claim_id))


def geocode_claim(conn, claim_id: int, locations: list[dict], article_text: str) -> dict:
    loc_candidates = _viable_locations(locations, max_n=MAX_NER_CANDIDATES)
    if not loc_candidates:
        _set_status(conn, claim_id, "no_geocode")
        return {"status": "no_geocode", "reason": "no_high_conf_location"}
    options: list[dict] = []
    raw_results: list[dict] = []
    nominatim_failures: list[dict] = []
    for loc_text in loc_candidates:
        try:
            results = nominatim_search(loc_text, limit=NOMINATIM_RESULTS_PER_LOC)
        except NominatimError as e:
            # Track the failure but keep trying other locations — partial success is
            # better than no geocode if at least one of the NER candidates resolves.
            nominatim_failures.append({"loc": loc_text, "kind": e.kind, "status": e.status})
            continue
        for r in results:
            options.append(
                {
                    "loc_text": loc_text,
                    "display_name": r.get("display_name") or "",
                    "type": r.get("type") or "",
                    "importance": float(r.get("importance", 0) or 0),
                }
            )
            raw_results.append(r)
    if not options:
        _set_status(conn, claim_id, "no_geocode")
        # If ALL Nominatim calls hard-failed, surface that — it's not "place doesn't exist",
        # it's "we couldn't reach the geocoder". Important distinction at benchmark scale.
        if nominatim_failures and len(nominatim_failures) == len(loc_candidates):
            return {
                "status": "no_geocode",
                "reason": "nominatim_all_failed",
                "tried": loc_candidates,
                "failures": nominatim_failures,
            }
        return {
            "status": "no_geocode",
            "reason": "nominatim_empty_for_all",
            "tried": loc_candidates,
            "failures": nominatim_failures,
        }
    try:
        chosen_idx = _claude_pick_most_specific(article_text, options)
    except ClaudeDisambiguationError as e:
        _set_status(conn, claim_id, "no_geocode")
        return {
            "status": "no_geocode",
            "reason": f"claude_disambiguate_{e.reason}",
            "tried": loc_candidates,
            "n_candidates": len(options),
        }
    if chosen_idx is None:
        _set_status(conn, claim_id, "no_geocode")
        return {"status": "no_geocode", "reason": "claude_rejected_all", "tried": loc_candidates}
    chosen = raw_results[chosen_idx]
    chosen_loc = options[chosen_idx]["loc_text"]
    wkt = _bbox_from_nominatim(chosen)
    if not wkt:
        _set_status(conn, claim_id, "no_geocode")
        return {"status": "no_geocode", "reason": "no_bbox_in_chosen", "loc": chosen_loc}
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE claims
            SET bbox = ST_GeomFromText(%s, 4326),
                admin_region = %s,
                geocode_score = %s,
                status = 'geocoded'
            WHERE id = %s
            """,
            (wkt, chosen.get("display_name"), float(chosen.get("importance", 0) or 0), claim_id),
        )
    return {
        "status": "geocoded",
        "loc": chosen_loc,
        "place": (chosen.get("display_name") or "")[:80],
        "importance": round(float(chosen.get("importance", 0) or 0), 3),
        "n_candidates": len(options),
    }


def run_geocode(limit: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = {"geocoded": 0, "no_geocode": 0, "error": 0}
    with db.connect() as conn:
        with conn.cursor() as cur:
            sql = (
                "SELECT id, locations, raw_text FROM claims "
                "WHERE status='extracted' ORDER BY id"
            )
            if limit:
                sql += f" LIMIT {int(limit)}"
            cur.execute(sql)
            rows = cur.fetchall()
        for claim_id, locations, raw_text in rows:
            try:
                res = geocode_claim(conn, claim_id, locations, raw_text or "")
            except Exception as e:
                log.exception("geocode.error", claim_id=claim_id)
                _set_status(conn, claim_id, "error")
                res = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:160]}"}
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            log.info("geocode.claim", claim_id=claim_id, **res)
            conn.commit()
    return counts
