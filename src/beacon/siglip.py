"""SigLIP-based vision capabilities — runs locally via transformers since the
google/siglip-* zero-shot image-classification route isn't exposed on the HF
Inference Providers free tier.

Provides both:
  - zero_shot_classify(image, labels) → HF task: Zero-Shot Image Classification
  - embed_image(image) / embed_text(text) → HF task: Visual Document Retrieval
    (used by tile_archive in pgvector for similar-image lookup)

Single model load, two tasks.
"""

from __future__ import annotations

from pathlib import Path

import structlog

log = structlog.get_logger()

# google/siglip-base-patch16-224 = 203M params, 768-dim embeddings.
# Bigger SigLIP-2 variants exist but base is fast on CPU.
SIGLIP_MODEL = "google/siglip-base-patch16-224"

# Default labels for satellite tile classification — covers the major
# land-cover / event categories we care about.
DEFAULT_TILE_LABELS: tuple[str, ...] = (
    "burned land or fire scar",
    "flooded area or standing water",
    "healthy forest or vegetation",
    "urban or built-up area",
    "water body",
    "agricultural field or cropland",
    "bare earth or desert",
    "snow-covered terrain",
    "cloud cover",
)

_pipeline_cache = None
_model_cache = None
_processor_cache = None


def _zsic_pipeline():
    """Lazy-load the zero-shot image classification pipeline."""
    global _pipeline_cache
    if _pipeline_cache is None:
        from transformers import pipeline

        log.info("siglip.loading_pipeline", model=SIGLIP_MODEL)
        _pipeline_cache = pipeline(
            "zero-shot-image-classification",
            model=SIGLIP_MODEL,
            device="cpu",
        )
    return _pipeline_cache


def _embed_pair():
    """Lazy-load model + processor for image/text embedding.

    transformers 5.x lazy-loads submodules and `from transformers import SiglipModel`
    can fail in some load orders. Import directly from the submodule to bypass the
    lazy loader.
    """
    global _model_cache, _processor_cache
    if _model_cache is None:
        from transformers.models.siglip.modeling_siglip import SiglipModel
        from transformers.models.siglip.processing_siglip import SiglipProcessor

        log.info("siglip.loading_model_for_embed", model=SIGLIP_MODEL)
        _processor_cache = SiglipProcessor.from_pretrained(SIGLIP_MODEL)
        _model_cache = SiglipModel.from_pretrained(SIGLIP_MODEL)
        _model_cache.eval()
    return _model_cache, _processor_cache


def zero_shot_classify(
    image_path: str,
    candidate_labels: tuple[str, ...] = DEFAULT_TILE_LABELS,
    *,
    top_k: int = 5,
) -> list[dict]:
    """HF: Zero-Shot Image Classification. Returns top-K [{label, score}] sorted desc."""
    pipe = _zsic_pipeline()
    out = pipe(image_path, candidate_labels=list(candidate_labels))
    ranked = [{"label": r["label"], "score": float(r["score"])} for r in out]
    ranked.sort(key=lambda r: -r["score"])
    return ranked[:top_k]


def _unwrap(x):
    """Some transformers versions return BaseModelOutputWithPooling from
    get_image_features/get_text_features instead of a raw tensor. Unwrap defensively."""
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if v is not None:
                return v
    return x


def embed_image(image_path: str) -> list[float]:
    """SigLIP image embedding (768-dim). For HF Visual Document Retrieval."""
    import torch
    from PIL import Image

    model, processor = _embed_pair()
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = _unwrap(model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype("float32").tolist()


def embed_text(text: str) -> list[float]:
    """SigLIP text embedding (768-dim). Lets us search the tile archive by text query."""
    import torch

    model, processor = _embed_pair()
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        feats = _unwrap(model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype("float32").tolist()
