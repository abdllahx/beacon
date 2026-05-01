"""Tests for beacon.imagery — bbox + colormap pure functions."""

import numpy as np

from beacon.imagery import _normalize_bbox, _value_to_rgb


def test_normalize_bbox_clamps_too_small():
    # bbox smaller than MIN_BBOX_DEG (0.02) gets expanded
    out = _normalize_bbox((0.0, 0.0, 0.001, 0.001))
    w, s, e, n = out
    assert (e - w) >= 0.02
    assert (n - s) >= 0.02


def test_normalize_bbox_clamps_too_large():
    # bbox larger than MAX_BBOX_DEG (1.0) gets shrunk around centroid
    out = _normalize_bbox((-90.0, -45.0, 90.0, 45.0))
    w, s, e, n = out
    assert (e - w) <= 1.0
    assert (n - s) <= 1.0
    # Centroid preserved
    assert abs(((w + e) / 2) - 0.0) < 1e-6
    assert abs(((s + n) / 2) - 0.0) < 1e-6


def test_value_to_rgb_low_high_bounds():
    arr = np.array([[-0.5, 0.0, 0.7]], dtype=np.float32)
    rgb = _value_to_rgb(arr, vmin=-0.5, vmax=0.7, invert=False)
    # Min value → red-ish (R=255, G low, B=0)
    r0, g0, b0 = rgb[0, 0]
    assert r0 == 255
    assert g0 < 128
    assert b0 == 0
    # Mid value → yellowish (R=255, G high, B=0)
    r1, g1, b1 = rgb[0, 1]
    assert r1 == 255
    assert g1 > 100
    # Max value → green-ish (R lower, G high, B has color)
    r2, g2, b2 = rgb[0, 2]
    assert g2 > 100
    assert r2 < 255  # red drops on the green side


def test_value_to_rgb_invert_flips_polarity():
    arr = np.array([[0.5]], dtype=np.float32)
    a = _value_to_rgb(arr, vmin=-0.5, vmax=0.5, invert=False)
    b = _value_to_rgb(arr, vmin=-0.5, vmax=0.5, invert=True)
    # 0.5 (max) without invert is green-ish; with invert it's red-ish
    assert a[0, 0, 0] < 200  # green-ish has lower red
    assert b[0, 0, 0] == 255  # invert → red
