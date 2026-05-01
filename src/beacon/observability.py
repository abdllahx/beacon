"""Langfuse tracing wrapper. No-op when LANGFUSE_* env vars are absent — safe to
import unconditionally. Decorate any function with @observe to capture spans
once Langfuse Cloud keys are wired in .env.

This keeps Beacon's call graph traceable across providers (Claude SDK, HF
Inference, local CPU models) in one timeline."""
from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def _enabled() -> bool:
    """Check Langfuse keys via env first, then fall back to pydantic Settings (.env).

    Also export to os.environ so the langfuse client (which reads env directly)
    picks them up when it's initialized inside the @observe decorator."""
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return True
    try:
        from beacon.config import get_settings
        s = get_settings()
        if s.langfuse_public_key and s.langfuse_secret_key:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key)
            os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
            return True
    except Exception:
        pass
    return False


# Eagerly resolve so the decorator factory below sees env vars at import time.
_KEYS_PRESENT = _enabled()


try:
    from langfuse import observe as _lf_observe  # type: ignore[import-not-found]
    _LF_AVAILABLE = True
except ImportError:
    _LF_AVAILABLE = False


def observe(name: str | None = None, *, as_type: str | None = None) -> Callable[[F], F]:
    """Trace a function call when Langfuse is configured; otherwise pass through.

    Usage:
        @observe(name="vision_vqa", as_type="generation")
        def call_claude(...): ...
    """
    if _LF_AVAILABLE and _KEYS_PRESENT:
        kwargs: dict = {}
        if name is not None:
            kwargs["name"] = name
        if as_type is not None:
            kwargs["as_type"] = as_type
        return _lf_observe(**kwargs)  # type: ignore[no-any-return]

    def _identity(fn: F) -> F:
        @wraps(fn)
        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return _identity


def trace_metadata(**kv) -> None:
    """Attach metadata to the current trace (no-op if Langfuse missing)."""
    if not (_LF_AVAILABLE and _enabled()):
        return
    try:
        from langfuse import get_client  # type: ignore[import-not-found]
        client = get_client()
        client.update_current_span(metadata=kv)
    except Exception:
        pass
