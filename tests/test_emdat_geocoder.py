"""Tests for beacon.emdat_geocoder — query-building pure functions
(no Nominatim calls, no DB)."""

from beacon.emdat_geocoder import (
    _build_query,
    _query_from_admin_units,
    _query_from_location,
    _split_location_candidates,
)


def test_split_drops_generic_words_and_parens():
    parts = _split_location_candidates("Whakatāne, Ōhope, and Thornton communities (Bay of Plenty)")
    assert "Whakatāne" in parts
    assert "Ōhope" in parts
    # "and Thornton communities" → drop "and" + "communities" → just "Thornton"
    assert any(p == "Thornton" for p in parts)


def test_split_handles_semicolons_and_short_tokens():
    parts = _split_location_candidates("AB; Toronto, ON")
    # "AB" is too short (< 3) — should be filtered
    assert "AB" not in parts
    assert "Toronto" in parts


def test_query_from_admin_units_prefers_most_specific_level():
    adm = [{"adm0_name": "Brazil", "adm1_name": "Rio de Janeiro state", "adm2_name": "Rio de Janeiro city"}]
    q = _query_from_admin_units(adm, "Brazil")
    # Most specific is adm2 → city name selected
    assert q.startswith("Rio de Janeiro city")


def test_query_from_location_prepends_country_when_missing():
    q = _query_from_location("Pernik", "Bulgaria")
    assert "Pernik" in q
    assert "Bulgaria" in q


def test_query_from_location_skips_country_if_already_present():
    q = _query_from_location("Athens, Greece", "Greece")
    # Should NOT double-add Greece
    assert q.count("Greece") == 1


def test_build_query_falls_back_through_tiers():
    # admin_units missing → location used
    result = _build_query(None, "Spain", "Sevilla")
    assert result is not None
    query, source = result
    assert "Sevilla" in query
    assert source == "location_nominatim"
    # everything missing → country only
    result = _build_query(None, "Italy", None)
    assert result is not None
    assert result[1] == "country_nominatim"
    # nothing → None
    assert _build_query(None, None, None) is None
