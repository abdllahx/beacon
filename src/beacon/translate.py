"""Multi-language translation of verification reports via Helsinki-NLP MarianMT.

Hits HF task #6 from the spec (Translation). MarianMT models are per-language-pair
(opus-mt-en-XX) which keeps each model small (~75M params) and fast through the
HF Inference Providers free tier. NLLB-200 was first choice for breadth (200 langs)
but is not currently exposed on the Inference Providers translation route.

Output is persisted into `verification_runs.translations` as a JSONB map
{"es": "...", "ar": "...", "fr": "..."} for the Streamlit UI's language toggle.
"""

import structlog
from huggingface_hub import InferenceClient

from beacon.config import get_settings

log = structlog.get_logger()

# Helsinki-NLP/opus-mt-en-XX MarianMT models, one per target language pair.
LANG_TO_MODEL = {
    "es": "Helsinki-NLP/opus-mt-en-es",  # Spanish
    "ar": "Helsinki-NLP/opus-mt-en-ar",  # Arabic
    "fr": "Helsinki-NLP/opus-mt-en-fr",  # French
}
DEFAULT_TARGET_LANGS: tuple[str, ...] = ("es", "ar", "fr")

# MarianMT supports ~512 tokens per call; chunk longer text by paragraph to avoid
# silent truncation of long reports.
MAX_CHARS_PER_CHUNK = 1500


def _client() -> InferenceClient:
    return InferenceClient(token=get_settings().hf_token)


def _chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split text into chunks <= max_chars at paragraph or sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    out: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                out.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                # Paragraph itself is too long — split by sentence.
                buf = ""
                running = ""
                for sent in p.split(". "):
                    candidate = (running + ". " + sent) if running else sent
                    if len(candidate) <= max_chars:
                        running = candidate
                    else:
                        if running:
                            out.append(running)
                        running = sent
                if running:
                    buf = running
    if buf:
        out.append(buf)
    return out


def translate(text: str, *, target_lang: str) -> str | None:
    """Translate a single text block to the target language. Returns None on failure."""
    if target_lang not in LANG_TO_MODEL:
        raise ValueError(f"Unsupported target_lang: {target_lang!r} (supported: {list(LANG_TO_MODEL)})")
    if not text or not text.strip():
        return ""
    client = _client()
    model = LANG_TO_MODEL[target_lang]
    parts: list[str] = []
    for chunk in _chunks(text):
        try:
            out = client.translation(chunk, model=model)
            parts.append(out.translation_text)
        except Exception as e:
            log.warning("translate.chunk_failed", model=model, error=str(e)[:160])
            return None
    return "\n\n".join(parts)


def translate_all(text: str, *, target_langs: tuple[str, ...] = DEFAULT_TARGET_LANGS) -> dict[str, str | None]:
    """Translate the same text into all target languages."""
    out: dict[str, str | None] = {}
    for lang in target_langs:
        log.info("translate.start", lang=lang, chars=len(text or ""))
        out[lang] = translate(text, target_lang=lang)
    return out
