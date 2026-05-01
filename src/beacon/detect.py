"""Object detection via facebook/detr-resnet-50 (HF task: Object Detection).

Honest caveat: DETR-ResNet-50 was trained on COCO (everyday objects at 1-10m
photographic scale). At Sentinel-2's 10m/pixel resolution most COCO classes
(cars, people, animals) are 0-1 pixel and therefore undetectable. This node
is wired in as the Object Detection capability per the spec; on actual high-
resolution aerial/drone input it produces meaningful bounding boxes.
We persist whatever DETR returns plus a `resolution_limit_note` so the
synthesizer and demo can render the result honestly.
"""

import structlog
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont

from beacon.config import get_settings

log = structlog.get_logger()

DETECT_MODEL = "facebook/detr-resnet-50"
DEFAULT_SCORE_THRESHOLD = 0.5


def detect_objects(
    image_path: str,
    out_overlay_path: str,
    *,
    model: str = DETECT_MODEL,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> dict | None:
    """Run DETR, render bounding boxes onto a copy of the image.
    Returns {model, n_objects, classes_count, detections, overlay_path,
    resolution_limit_note} or None on failure."""
    client = InferenceClient(token=get_settings().hf_token)
    try:
        out = client.object_detection(image_path, model=model)
    except Exception as e:
        log.warning("detect.failed", error=str(e)[:160])
        return None

    detections: list[dict] = []
    classes_count: dict[str, int] = {}
    for d in out:
        label = getattr(d, "label", None)
        score = float(getattr(d, "score", 0))
        box = getattr(d, "box", None)
        if score < score_threshold or not label or not box:
            continue
        # box is BoundingBox (xmin, ymin, xmax, ymax)
        bb = {
            "xmin": int(getattr(box, "xmin", 0)),
            "ymin": int(getattr(box, "ymin", 0)),
            "xmax": int(getattr(box, "xmax", 0)),
            "ymax": int(getattr(box, "ymax", 0)),
        }
        detections.append({"label": label, "score": round(score, 3), "box": bb})
        classes_count[label] = classes_count.get(label, 0) + 1

    # Render overlay even if 0 detections (blank passthrough so we always have an image)
    img = Image.open(image_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in detections:
        b = d["box"]
        draw.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=(255, 60, 0), width=3)
        label_text = f"{d['label']} {d['score']:.2f}"
        if font is not None:
            draw.text((b["xmin"] + 2, max(0, b["ymin"] - 12)), label_text, fill=(255, 60, 0), font=font)
        else:
            draw.text((b["xmin"] + 2, max(0, b["ymin"] - 12)), label_text, fill=(255, 60, 0))
    img.save(out_overlay_path)

    return {
        "model": model,
        "n_objects": len(detections),
        "classes_count": classes_count,
        "detections": detections[:50],  # cap to avoid huge JSONB blobs
        "overlay_path": out_overlay_path,
        "resolution_limit_note": (
            "DETR-ResNet-50 is COCO-trained; at Sentinel-2's 10m/pixel most COCO classes "
            "are sub-pixel and therefore undetectable. High-resolution aerial / drone input "
            "would produce dense detections at the same node."
        ),
    }
