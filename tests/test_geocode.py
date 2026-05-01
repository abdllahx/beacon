"""Tests for beacon.geocode — typed errors and pure helpers."""

from beacon.geocode import (
    ClaudeDisambiguationError,
    NominatimError,
    _bbox_from_nominatim,
    _bbox_wkt,
    _viable_locations,
)


def test_bbox_wkt_polygon_format():
    wkt = _bbox_wkt(34.0, 35.0, -119.0, -118.0)
    # PostGIS polygon, 5 vertices (closed ring), lon-lat order
    assert wkt.startswith("POLYGON((")
    assert "-119.0 34.0" in wkt
    assert "-118.0 35.0" in wkt


def test_bbox_from_nominatim_handles_valid_and_invalid():
    valid = {"boundingbox": ["34.0", "35.0", "-119.0", "-118.0"]}
    assert _bbox_from_nominatim(valid) is not None
    assert _bbox_from_nominatim({}) is None
    assert _bbox_from_nominatim({"boundingbox": ["bad"]}) is None
    assert _bbox_from_nominatim({"boundingbox": ["nope", "x", "y", "z"]}) is None


def test_viable_locations_filters_low_score_and_short_text():
    locs = [
        {"text": "X", "score": 0.99},  # too short
        {"text": "Mexico", "score": 0.5},  # below MIN_NER_SCORE
        {"text": "Pernik", "score": 0.95},
        {"text": "Pernik", "score": 0.97},  # duplicate (case-insensitive)
        {"text": "Sofia", "score": 0.92},
    ]
    out = _viable_locations(locs, max_n=3)
    assert "Pernik" in out
    assert "Sofia" in out
    assert "X" not in out
    assert "Mexico" not in out
    # Duplicate dedup
    assert out.count("Pernik") == 1


def test_typed_errors_carry_context():
    e = NominatimError("rate_limited", status=429, query="foo", detail="too many")
    assert e.kind == "rate_limited"
    assert e.status == 429
    assert "rate_limited" in str(e)

    e2 = ClaudeDisambiguationError(reason="exception", detail="boom")
    assert e2.reason == "exception"
    assert "exception" in str(e2)
