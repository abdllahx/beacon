"""Land-cover segmentation via NVIDIA SegFormer-B0 on ADE20K.

Hits HF task #7 — Image Segmentation. ADE20K has 150 classes including the
natural-scene categories most relevant for satellite imagery (tree, grass,
earth, water, mountain, sky, sand, road, building, sea, lake, river).

We run on the AFTER Sentinel-2 tile and produce:
  1. A colored overlay PNG (one color per top class) for the demo
  2. Per-class area percentages, persisted into imagery_metadata.s2.after.segmentation

Combined with dNBR's burn-pct, this gives the synthesizer rich signal:
"X% of the after-tile is now bare earth where it was forest in the before-tile."
"""

import base64
import io

import numpy as np
import structlog
from huggingface_hub import InferenceClient
from PIL import Image

from beacon.config import get_settings

log = structlog.get_logger()

SEGMENT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

# Hand-picked color palette for the most common land-cover classes.
# Other classes get a deterministic hash-derived color.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "tree": (34, 139, 34),
    "grass": (124, 252, 0),
    "earth": (139, 90, 43),
    "water": (30, 144, 255),
    "sea": (30, 144, 255),
    "lake": (30, 144, 255),
    "river": (30, 144, 255),
    "mountain": (139, 69, 19),
    "sky": (135, 206, 235),
    "sand": (238, 214, 175),
    "rock": (105, 105, 105),
    "road": (50, 50, 50),
    "building": (220, 20, 60),
    "field": (189, 183, 107),
}


def _color_for(label: str) -> tuple[int, int, int]:
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    h = abs(hash(label)) % 0xFFFFFF
    return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)


def _decode_mask(mask_obj) -> np.ndarray:
    """HF returns each segment's mask as a PIL.Image, base64 string, or dict.
    Returns a 2D bool array."""
    if isinstance(mask_obj, Image.Image):
        return np.array(mask_obj.convert("L")) > 127
    if isinstance(mask_obj, str):
        # Base64-encoded PNG
        data = base64.b64decode(mask_obj.split(",", 1)[-1])
        img = Image.open(io.BytesIO(data)).convert("L")
        return np.array(img) > 127
    if isinstance(mask_obj, np.ndarray):
        return mask_obj > 0
    raise TypeError(f"Unexpected mask type: {type(mask_obj)}")


def segment_image(
    image_path: str,
    out_overlay_path: str,
    *,
    model: str = SEGMENT_MODEL,
    overlay_alpha: float = 0.5,
) -> dict | None:
    """Run SegFormer on `image_path`, save colored overlay to `out_overlay_path`,
    return {classes: {label: area_pct}, top: [...], mask_path: ..., model: ...}."""
    client = InferenceClient(token=get_settings().hf_token)
    try:
        segments = client.image_segmentation(image_path, model=model)
    except Exception as e:
        log.warning("segment.failed", error=str(e)[:160])
        return None
    if not segments:
        return None

    base = Image.open(image_path).convert("RGB")
    H, W = base.height, base.width
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    counts: dict[str, int] = {}

    for seg in segments:
        label = getattr(seg, "label", None) or (seg.get("label") if isinstance(seg, dict) else None)
        mask_obj = getattr(seg, "mask", None) or (seg.get("mask") if isinstance(seg, dict) else None)
        if not label or mask_obj is None:
            continue
        try:
            mask = _decode_mask(mask_obj)
        except Exception as e:
            log.warning("segment.mask_decode_failed", label=label, error=str(e)[:120])
            continue
        if mask.shape != (H, W):
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)
            mask = np.array(mask_img) > 127
        color = np.array(_color_for(label), dtype=np.uint8)
        overlay[mask] = color
        counts[label] = counts.get(label, 0) + int(mask.sum())

    total_px = H * W
    classes_pct = {label: round(100.0 * n / total_px, 2) for label, n in counts.items()}
    # Compose overlay onto base image
    base_arr = np.array(base, dtype=np.uint8)
    blended = np.where(
        overlay.sum(axis=-1, keepdims=True) > 0,
        (base_arr * (1 - overlay_alpha) + overlay * overlay_alpha).astype(np.uint8),
        base_arr,
    )
    Image.fromarray(blended).save(out_overlay_path)
    top_classes = sorted(classes_pct.items(), key=lambda kv: -kv[1])[:5]
    return {
        "model": model,
        "mask_path": out_overlay_path,
        "classes_pct": classes_pct,
        "top": [{"label": l, "pct": p} for l, p in top_classes],
    }
