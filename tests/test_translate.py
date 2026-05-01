"""Tests for beacon.translate — chunking pure function (no API calls)."""

from beacon.translate import LANG_TO_MODEL, MAX_CHARS_PER_CHUNK, _chunks


def test_chunks_short_text_is_single_chunk():
    out = _chunks("Hello world.")
    assert out == ["Hello world."]


def test_chunks_long_text_splits_at_paragraph_boundary():
    para = "A " * 600  # 1200 chars
    text = para + "\n\n" + para + "\n\n" + para
    chunks = _chunks(text, max_chars=MAX_CHARS_PER_CHUNK)
    assert len(chunks) >= 2
    # Each chunk fits within limit (with mild slack for the last chunk)
    for c in chunks:
        assert len(c) <= MAX_CHARS_PER_CHUNK + 2  # +2 for paragraph join


def test_chunks_handles_paragraph_too_long_via_sentence_split():
    # Single paragraph > max_chars must fall back to sentence-level split
    sent = "This is a sentence. " * 200  # 4000 chars, no paragraph break
    chunks = _chunks(sent, max_chars=1500)
    # No sentence is itself > 1500 chars, so all chunks should fit
    for c in chunks:
        assert len(c) <= 1500


def test_supported_languages_have_marian_models():
    for lang in ("es", "ar", "fr"):
        assert lang in LANG_TO_MODEL
        assert LANG_TO_MODEL[lang].startswith("Helsinki-NLP/opus-mt-en-")
