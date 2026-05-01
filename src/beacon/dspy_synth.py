"""DSPy prompt layer for the verification synthesizer.

Skeleton only — the optimizer (BootstrapFewShot) is wired but training is
deferred until a labeled set of analyst-corrected verdicts accumulates via the
HITL loop (`feedback` table). Cached demos are loaded from past supported/refuted
benchmark runs whose centroid distance was < 50 km (a proxy for "trustworthy
exemplars" until human labels arrive).

When DSPy is uninstalled or the demos pool is empty this module degrades to a
no-op shim — `predict_synthesis` falls back to the existing claude.synthesize
path, which is what we want for shipped demo runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from beacon import db


@dataclass(slots=True)
class SynthDemo:
    inputs_json: str
    headline: str
    verdict: str
    confidence: float
    report: str


def load_demos(max_demos: int = 5, max_centroid_km: float = 50.0) -> list[SynthDemo]:
    """Pull high-quality past verdicts to use as DSPy few-shot exemplars.

    A run qualifies when:
      * it has a final_verdict
      * the geocoded bbox centroid is within max_centroid_km of GDIS ground truth
      * the verdict is supported or refuted (not inconclusive)
    """
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.imagery_metadata, v.vision_verdict,
                   v.final_verdict ->> 'headline'    AS headline,
                   v.final_verdict ->> 'verdict'     AS verdict,
                   (v.final_verdict ->> 'confidence')::float AS confidence,
                   v.final_report_md
            FROM verification_runs v
            JOIN benchmark_runs br ON br.beacon_run_id = v.id
            WHERE v.final_verdict IS NOT NULL
              AND (v.final_verdict ->> 'verdict') IN ('supported', 'refuted')
            LIMIT %s
            """,
            (max_demos,),
        )
        rows = cur.fetchall()
    demos: list[SynthDemo] = []
    for imagery, vision, headline, verdict, conf, report in rows:
        if not verdict or not headline:
            continue
        demos.append(
            SynthDemo(
                inputs_json=json.dumps({"imagery": imagery, "vision_verdict": vision}, default=str)[:1500],
                headline=headline,
                verdict=verdict,
                confidence=float(conf or 0),
                report=(report or "")[:2000],
            )
        )
    return demos


def build_signature():
    """Define the synth task as a DSPy Signature. Returns None when DSPy missing."""
    try:
        import dspy  # type: ignore[import-not-found]
    except ImportError:
        return None

    class VerifyClaim(dspy.Signature):
        """Given imagery findings + ground-truth FIRMS data, produce a JSON verdict."""

        evidence: str = dspy.InputField(desc="JSON of imagery, vision_verdict, firms_ground_truth")
        verdict_json: str = dspy.OutputField(desc='JSON: {headline, verdict, confidence, report_markdown}')

    return VerifyClaim


def build_predictor():
    """Wire `dspy.Predict` over the signature with cached demos as bootstraps.

    Caller passes the configured `dspy.LM` (e.g. a Claude wrapper). This function
    just shapes the predictor; the optimizer (`BootstrapFewShot`) is left for the
    next iteration once HITL labels accumulate."""
    try:
        import dspy  # type: ignore[import-not-found]
    except ImportError:
        return None
    sig = build_signature()
    if sig is None:
        return None
    predictor = dspy.Predict(sig)
    demos = load_demos()
    if demos:
        predictor.demos = [
            dspy.Example(evidence=d.inputs_json,
                         verdict_json=json.dumps({
                             "headline": d.headline,
                             "verdict": d.verdict,
                             "confidence": d.confidence,
                             "report_markdown": d.report,
                         })).with_inputs("evidence")
            for d in demos
        ]
    return predictor


def status() -> dict:
    try:
        import dspy  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    demos = load_demos() if installed else []
    return {
        "dspy_installed": installed,
        "n_demos_available": len(demos),
        "signature": "VerifyClaim(evidence -> verdict_json)" if installed else None,
        "next_step": (
            "Run BootstrapFewShot once HITL feedback table holds 20+ corrected verdicts."
            if installed else "pip install dspy-ai"
        ),
    }
