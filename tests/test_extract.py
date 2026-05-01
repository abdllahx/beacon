"""Tests for beacon.extract — pure-function bits that don't hit HF."""

from beacon.extract import (
    HAZARD_CATEGORIES,
    KEEP_LABELS,
    LABEL_TO_EVENT_TYPE,
    _bucket_entities,
)


def test_bucket_entities_separates_loc_per_actor_dates():
    ents = [
        {"text": "California", "score": 0.99, "start": 0, "end": 10, "entity_group": "LOC"},
        {"text": "Joe", "score": 0.95, "start": 12, "end": 15, "entity_group": "PER"},
        {"text": "FEMA", "score": 0.95, "start": 20, "end": 24, "entity_group": "ORG"},
        {"text": "May 2025", "score": 0.90, "start": 30, "end": 38, "entity_group": "DATE"},
        {"text": "MISC_TOKEN", "score": 0.6, "start": 40, "end": 50, "entity_group": "MISC"},
    ]
    locs, dates, actors = _bucket_entities(ents)
    assert [loc["text"] for loc in locs] == ["California"]
    assert [d["text"] for d in dates] == ["May 2025"]
    assert {a["text"] for a in actors} == {"Joe", "FEMA"}


def test_bucket_entities_sorts_locations_by_score_descending():
    ents = [
        {"text": "Smaller", "score": 0.7, "start": 0, "end": 7, "entity_group": "LOC"},
        {"text": "Bigger", "score": 0.99, "start": 10, "end": 16, "entity_group": "LOC"},
    ]
    locs, _, _ = _bucket_entities(ents)
    assert locs[0]["text"] == "Bigger"
    assert locs[1]["text"] == "Smaller"


def test_hazard_categories_keep_six_event_types():
    keep = [c for c in HAZARD_CATEGORIES if c[2]]
    assert len(keep) == 6
    event_types = {c[1] for c in keep}
    assert {"wildfire", "flood", "storm", "earthquake", "landslide", "volcanic"} == event_types


def test_label_to_event_type_round_trip():
    for label, event_type, keep in HAZARD_CATEGORIES:
        if keep:
            assert LABEL_TO_EVENT_TYPE[label] == event_type
        if keep:
            assert label in KEEP_LABELS
