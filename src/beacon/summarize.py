"""Extractive summarization via facebook/bart-large-cnn (HF task: Summarization).

Runs alongside the Claude-narrated synthesizer to produce a short, BART-distilled
summary of the article + claim — useful for cards / list views and demonstrably
hits the Summarization HF task end-to-end via Inference Providers.
"""

import structlog
from huggingface_hub import InferenceClient

from beacon.config import get_settings

log = structlog.get_logger()

SUMMARIZE_MODEL = "facebook/bart-large-cnn"
DEFAULT_MAX_LENGTH = 80
DEFAULT_MIN_LENGTH = 25


def _client() -> InferenceClient:
    return InferenceClient(token=get_settings().hf_token)


def summarize(text: str, *, max_length: int = DEFAULT_MAX_LENGTH, min_length: int = DEFAULT_MIN_LENGTH) -> str | None:
    """One-shot extractive summary. Returns None on failure."""
    if not text or not text.strip():
        return None
    # BART-CNN was trained on CNN/DailyMail with input length up to ~1024 tokens.
    # Truncate to a safe character budget to avoid silent server-side cutoff.
    trimmed = text[:3500]
    try:
        out = _client().summarization(trimmed, model=SUMMARIZE_MODEL)
    except Exception as e:
        log.warning("summarize.failed", error=str(e)[:160])
        return None
    text_out = getattr(out, "summary_text", None)
    if text_out is None and isinstance(out, dict):
        text_out = out.get("summary_text")
    return text_out
