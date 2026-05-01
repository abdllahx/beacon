from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
import structlog
from PIL import Image
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

log = structlog.get_logger()

from beacon.tunables import (
    DEFAULT_TILE_SIZE_PX,
    DNBR_BURN_PCT_THRESHOLD,
    MAX_BBOX_DEG,
    MIN_BBOX_DEG,
    S1_DECREASE_THRESHOLD_DB,
)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"
TILE_DIR = Path("data/tiles")


def _normalize_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, s, e, n = bbox
    cx, cy = (w + e) / 2, (s + n) / 2
    half_w = max(MIN_BBOX_DEG / 2, min(MAX_BBOX_DEG / 2, (e - w) / 2))
    half_h = max(MIN_BBOX_DEG / 2, min(MAX_BBOX_DEG / 2, (n - s) / 2))
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _stac_client() -> Client:
    return Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)


def find_best_item(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    *,
    collection: str = DEFAULT_COLLECTION,
    max_items: int = 50,
):
    """Pick a Sentinel-2 item that (a) actually covers the bbox and (b) has the lowest cloud.

    The default STAC bbox filter only requires *intersection*, which can return scenes that
    cover < 5% of the target. We re-rank by intersection-over-target area first, then cloud.
    """
    from shapely.geometry import box, shape

    client = _stac_client()
    target = box(*bbox)
    search = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        max_items=max_items,
    )
    items = list(search.items())
    if not items:
        return None
    scored: list[tuple] = []
    for item in items:
        try:
            geom = shape(item.geometry)
            coverage = geom.intersection(target).area / max(target.area, 1e-9)
        except Exception:
            coverage = 0.0
        cloud = float(item.properties.get("eo:cloud_cover", 100) or 100)
        scored.append((item, coverage, cloud))
    for cov_thresh in (0.95, 0.7, 0.5):
        cands = [s for s in scored if s[1] >= cov_thresh]
        if cands:
            cands.sort(key=lambda s: s[2])
            return cands[0][0]
    scored.sort(key=lambda s: (s[1] * 100 - s[2] * 0.5), reverse=True)
    return scored[0][0]


def fetch_tile(
    bbox: tuple[float, float, float, float],
    date_center: datetime,
    *,
    window_days: int,
    out_path: Path,
    size_px: int = DEFAULT_TILE_SIZE_PX,
) -> dict | None:
    """Find best Sentinel-2 visual tile near date_center, crop to bbox, save as PNG.

    window_days: positive → look forward from date_center; negative → look backward.
    """
    bbox = _normalize_bbox(bbox)
    if window_days >= 0:
        start, end = date_center, date_center + timedelta(days=window_days)
    else:
        start, end = date_center + timedelta(days=window_days), date_center
    item = find_best_item(bbox, start, end)
    if item is None:
        log.info("imagery.no_item", bbox=bbox, start=start.isoformat(), end=end.isoformat())
        return None
    visual = item.assets.get("visual")
    if visual is None:
        log.warning("imagery.no_visual_asset", item_id=item.id)
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(visual.href) as src:
        bounds_in_src = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        window = from_bounds(*bounds_in_src, transform=src.transform)
        ratio = window.width / max(window.height, 1)
        if ratio >= 1:
            out_w, out_h = size_px, max(64, int(size_px / ratio))
        else:
            out_h, out_w = size_px, max(64, int(size_px * ratio))
        arr = src.read(
            indexes=[1, 2, 3],
            window=window,
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    img.save(out_path)
    return {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "path": str(out_path),
        "size_px": [out_w, out_h],
    }


def fetch_before_after_for_claim(
    claim_id: int,
    bbox: tuple[float, float, float, float],
    event_date: datetime,
    *,
    before_days: int = 30,
    after_days: int = 14,
    size_px: int = DEFAULT_TILE_SIZE_PX,
) -> dict:
    before_path = TILE_DIR / f"claim_{claim_id}_before.png"
    after_path = TILE_DIR / f"claim_{claim_id}_after.png"
    before = fetch_tile(bbox, event_date, window_days=-before_days, out_path=before_path, size_px=size_px)
    after = fetch_tile(bbox, event_date, window_days=after_days, out_path=after_path, size_px=size_px)
    return {"before": before, "after": after}


# ---------------------------------------------------------------------------
# NBR (Normalized Burn Ratio) composites — Phase B1
# ---------------------------------------------------------------------------
#
# NBR = (NIR - SWIR2) / (NIR + SWIR2) using Sentinel-2 bands B08 (NIR, 10m) and
# B12 (SWIR2, 20m). Burned ground reflects strongly in SWIR and weakly in NIR,
# so its NBR drops sharply. dNBR = NBR_before - NBR_after highlights the burn.
#
# This is what fixes the 3 "inconclusive" forest events from M1: TCI can't tell
# burned conifer canopy from healthy dark conifer canopy — NBR can.


def _read_band_to_window(
    asset_href: str,
    bbox: tuple[float, float, float, float],
    *,
    size_px: int,
) -> np.ndarray:
    """Read a single Sentinel-2 band, cropped + resampled to the bbox window."""
    with rasterio.open(asset_href) as src:
        bounds_in_src = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        window = from_bounds(*bounds_in_src, transform=src.transform)
        ratio = window.width / max(window.height, 1)
        if ratio >= 1:
            out_w, out_h = size_px, max(64, int(size_px / ratio))
        else:
            out_h, out_w = size_px, max(64, int(size_px * ratio))
        arr = src.read(
            indexes=[1],
            window=window,
            out_shape=(1, out_h, out_w),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
    return arr[0].astype(np.float32)


def _value_to_rgb(arr: np.ndarray, *, vmin: float, vmax: float, invert: bool) -> np.ndarray:
    """Diverging red→yellow→green colormap. invert=True flips the polarity."""
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    if invert:
        norm = 1 - norm
    # 0.0 = red (255,0,0), 0.5 = yellow (255,220,0), 1.0 = green (40,180,60)
    r = np.where(norm < 0.5, 255, (255 - (norm - 0.5) * 2 * 215)).clip(0, 255).astype(np.uint8)
    g = np.where(norm < 0.5, (norm * 2 * 220), (220 - (norm - 0.5) * 2 * 40)).clip(0, 255).astype(np.uint8)
    b = np.where(norm < 0.5, 0, ((norm - 0.5) * 2 * 60)).clip(0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def fetch_nbr_tile(
    bbox: tuple[float, float, float, float],
    date_center: datetime,
    *,
    window_days: int,
    out_path: Path,
    size_px: int = DEFAULT_TILE_SIZE_PX,
) -> dict | None:
    """Compute NBR from B08 + B12 for the best-available Sentinel-2 scene.

    Saves a colored PNG (red=burned/bare, green=healthy vegetation) plus the raw
    float NBR array as `.npy` next to the PNG for downstream dNBR computation.
    """
    bbox = _normalize_bbox(bbox)
    if window_days >= 0:
        start, end = date_center, date_center + timedelta(days=window_days)
    else:
        start, end = date_center + timedelta(days=window_days), date_center
    item = find_best_item(bbox, start, end)
    if item is None:
        log.info("imagery.nbr.no_item", bbox=bbox)
        return None
    b08 = item.assets.get("B08")
    b12 = item.assets.get("B12")
    if not b08 or not b12:
        log.warning("imagery.nbr.missing_assets", item_id=item.id, has_b08=bool(b08), has_b12=bool(b12))
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nir = _read_band_to_window(b08.href, bbox, size_px=size_px)
    swir = _read_band_to_window(b12.href, bbox, size_px=size_px)
    eps = 1e-6
    nbr = (nir - swir) / (nir + swir + eps)
    nbr = np.clip(nbr, -1.0, 1.0)
    rgb = _value_to_rgb(nbr, vmin=-0.5, vmax=0.7, invert=False)
    Image.fromarray(rgb).save(out_path)
    np.save(out_path.with_suffix(".npy"), nbr.astype(np.float32))
    return {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "path": str(out_path),
        "array_path": str(out_path.with_suffix(".npy")),
    }


def compute_dnbr(
    before_array_path: Path,
    after_array_path: Path,
    out_path: Path,
    *,
    burn_threshold: float = DNBR_BURN_PCT_THRESHOLD,
) -> dict | None:
    """dNBR = NBR_before - NBR_after. Positive values = burned. Renders a colored
    heatmap PNG and reports the percentage of pixels exceeding `burn_threshold`
    (Key & Benson 'moderate' burn severity)."""
    if not before_array_path.exists() or not after_array_path.exists():
        return None
    nbr_before = np.load(before_array_path)
    nbr_after = np.load(after_array_path)
    # Resize after to before's shape if they differ (different scenes can have different framings).
    if nbr_before.shape != nbr_after.shape:
        from PIL import Image as _Image

        ar = _Image.fromarray(nbr_after).resize(
            (nbr_before.shape[1], nbr_before.shape[0]), _Image.BILINEAR
        )
        nbr_after = np.array(ar, dtype=np.float32)
    dnbr = (nbr_before - nbr_after).astype(np.float32)
    rgb = _value_to_rgb(dnbr, vmin=-0.2, vmax=0.6, invert=True)
    Image.fromarray(rgb).save(out_path)
    np.save(out_path.with_suffix(".npy"), dnbr)
    burn_mask = dnbr > burn_threshold
    burn_pct = float(burn_mask.mean()) * 100.0
    return {
        "path": str(out_path),
        "array_path": str(out_path.with_suffix(".npy")),
        "burn_pct": round(burn_pct, 2),
        "burn_threshold": burn_threshold,
        "shape": list(dnbr.shape),
    }


# ---------------------------------------------------------------------------
# Sentinel-1 SAR (Synthetic Aperture Radar) — Phase B4
# ---------------------------------------------------------------------------
# C-band radar penetrates clouds and works day/night. Backscatter behavior:
#   - Open water: very low (specular reflection away from sensor) — floods darken
#   - Forest canopy: moderate (volume scattering in branches)
#   - Burned/cleared ground: lower than intact canopy
#   - Built-up: high (corner reflectors in buildings)
# We use the Microsoft Planetary Computer's "sentinel-1-rtc" collection (Radiometric
# Terrain Corrected). Change detection = AFTER_dB - BEFORE_dB; large negative deltas
# flag floods and burns; large positive deltas flag new construction.
S1_COLLECTION = "sentinel-1-rtc"


def find_best_s1_item(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    *,
    max_items: int = 30,
):
    from shapely.geometry import box, shape

    client = _stac_client()
    target = box(*bbox)
    search = client.search(
        collections=[S1_COLLECTION],
        bbox=list(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        max_items=max_items,
    )
    items = list(search.items())
    if not items:
        return None
    scored = []
    for item in items:
        try:
            geom = shape(item.geometry)
            coverage = geom.intersection(target).area / max(target.area, 1e-9)
        except Exception:
            coverage = 0.0
        scored.append((item, coverage))
    for cov_thresh in (0.95, 0.7, 0.5):
        cands = [s for s in scored if s[1] >= cov_thresh]
        if cands:
            cands.sort(key=lambda s: -s[1])
            return cands[0][0]
    scored.sort(key=lambda s: -s[1])
    return scored[0][0]


def fetch_s1_tile(
    bbox: tuple[float, float, float, float],
    date_center: datetime,
    *,
    window_days: int,
    out_path: Path,
    size_px: int = DEFAULT_TILE_SIZE_PX,
) -> dict | None:
    """Fetch Sentinel-1 RTC VV polarization, render as a grayscale dB PNG."""
    bbox = _normalize_bbox(bbox)
    if window_days >= 0:
        start, end = date_center, date_center + timedelta(days=window_days)
    else:
        start, end = date_center + timedelta(days=window_days), date_center
    item = find_best_s1_item(bbox, start, end)
    if item is None:
        log.info("imagery.s1.no_item", bbox=bbox)
        return None
    vv = item.assets.get("vv")
    if not vv:
        log.warning("imagery.s1.no_vv", item_id=item.id)
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vv_arr = _read_band_to_window(vv.href, bbox, size_px=size_px)
    eps = 1e-10
    vv_db = 10 * np.log10(np.maximum(vv_arr, eps))
    np.save(out_path.with_suffix(".npy"), vv_db.astype(np.float32))
    # Display: stretch land-typical -25..+5 dB to 0..255
    img = np.clip((vv_db + 25) / 30, 0, 1)
    img8 = (img * 255).astype(np.uint8)
    rgb = np.stack([img8, img8, img8], axis=-1)
    Image.fromarray(rgb).save(out_path)
    return {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "polarization": "vv",
        "path": str(out_path),
        "array_path": str(out_path.with_suffix(".npy")),
    }


def compute_s1_change(
    before_array_path: Path,
    after_array_path: Path,
    out_path: Path,
    *,
    decrease_threshold_db: float = S1_DECREASE_THRESHOLD_DB,
) -> dict | None:
    """log-ratio change in dB. Negative deltas = darker after (flood/burn); positive = brighter (new build)."""
    if not before_array_path.exists() or not after_array_path.exists():
        return None
    before = np.load(before_array_path)
    after = np.load(after_array_path)
    if before.shape != after.shape:
        from PIL import Image as _Image

        ar = _Image.fromarray(after).resize(
            (before.shape[1], before.shape[0]), _Image.BILINEAR
        )
        after = np.array(ar, dtype=np.float32)
    delta = (after - before).astype(np.float32)
    rgb = _value_to_rgb(delta, vmin=-6.0, vmax=6.0, invert=True)
    Image.fromarray(rgb).save(out_path)
    np.save(out_path.with_suffix(".npy"), delta)
    decrease_pct = float((delta < -decrease_threshold_db).mean()) * 100.0
    return {
        "path": str(out_path),
        "array_path": str(out_path.with_suffix(".npy")),
        "decrease_pct": round(decrease_pct, 2),
        "decrease_threshold_db": decrease_threshold_db,
        "shape": list(delta.shape),
    }


def fetch_nbr_pair_for_claim(
    claim_id: int,
    bbox: tuple[float, float, float, float],
    event_date: datetime,
    *,
    before_days: int = 30,
    after_days: int = 14,
    size_px: int = DEFAULT_TILE_SIZE_PX,
) -> dict:
    before_path = TILE_DIR / f"claim_{claim_id}_nbr_before.png"
    after_path = TILE_DIR / f"claim_{claim_id}_nbr_after.png"
    dnbr_path = TILE_DIR / f"claim_{claim_id}_dnbr.png"
    before = fetch_nbr_tile(bbox, event_date, window_days=-before_days, out_path=before_path, size_px=size_px)
    after = fetch_nbr_tile(bbox, event_date, window_days=after_days, out_path=after_path, size_px=size_px)
    dnbr = None
    if before and after:
        dnbr = compute_dnbr(
            Path(before["array_path"]),
            Path(after["array_path"]),
            dnbr_path,
        )
    return {"nbr_before": before, "nbr_after": after, "dnbr": dnbr}
