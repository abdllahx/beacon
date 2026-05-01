"""Tests for beacon.eval_metrics — pure-function correctness checks
that don't depend on the DB or HF endpoints."""

import math

from beacon.eval_metrics import ACCURACY_THRESHOLDS_KM, haversine_km


def test_haversine_zero_distance():
    assert haversine_km(0, 0, 0, 0) == 0.0


def test_haversine_known_pair_paris_london():
    # Paris ~ (48.8566, 2.3522), London ~ (51.5074, -0.1278). Real-world ~344 km.
    d = haversine_km(48.8566, 2.3522, 51.5074, -0.1278)
    assert 340 < d < 350


def test_haversine_symmetric():
    a = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)  # NYC → LA
    b = haversine_km(34.0522, -118.2437, 40.7128, -74.0060)
    assert math.isclose(a, b, rel_tol=1e-9)


def test_haversine_antipodes():
    # Antipodes should be ~half Earth's circumference, ~20003 km.
    d = haversine_km(0, 0, 0, 180)
    assert 20000 < d < 20100


def test_accuracy_thresholds_are_geoparsing_literature():
    # The default thresholds are the convention from Gritta et al. 2018.
    assert ACCURACY_THRESHOLDS_KM == (10, 50, 161)
