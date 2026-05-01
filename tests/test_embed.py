"""Tests for beacon.embed and beacon.vdr — vector serialization."""

from beacon.embed import EMBED_DIM, _vector_literal as embed_literal
from beacon.vdr import _vector_literal as vdr_literal


def test_embed_dim_is_minilm():
    # all-MiniLM-L6-v2 produces 384-dim embeddings.
    assert EMBED_DIM == 384


def test_vector_literal_format_for_pgvector():
    vec = [0.1, -0.2, 0.333333]
    s = embed_literal(vec)
    assert s.startswith("[")
    assert s.endswith("]")
    assert "0.100000" in s
    assert "-0.200000" in s


def test_vector_literal_is_pgvector_parseable_shape():
    # pgvector accepts: '[a, b, c]' as a text literal
    s = embed_literal([1.0, 2.0, 3.0])
    assert s == "[1.000000,2.000000,3.000000]"


def test_vdr_literal_matches_embed_literal_shape():
    # Both modules render in the same pgvector text format.
    vec = [0.5] * 8
    assert embed_literal(vec) == vdr_literal(vec)
