"""Latency analytics over verification_runs timestamps.

Reads existing started_at / completed_at columns — no new pipeline runs needed.
Reports p50 / p95 / p99 in seconds for resume-bullet defensibility."""
from __future__ import annotations

from beacon import db


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def latency_stats() -> dict:
    """Return p50/p95/p99 + per-status counts over completed runs."""
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXTRACT(EPOCH FROM (completed_at - started_at)) AS secs,
                   status
            FROM verification_runs
            WHERE started_at IS NOT NULL
              AND completed_at IS NOT NULL
              AND completed_at > started_at
            """
        )
        rows = cur.fetchall()
    secs = sorted(float(r[0]) for r in rows if r[0] is not None and r[0] > 0)
    by_status: dict[str, int] = {}
    for _, st in rows:
        by_status[st] = by_status.get(st, 0) + 1
    return {
        "n": len(secs),
        "p50_sec": round(percentile(secs, 50), 1),
        "p95_sec": round(percentile(secs, 95), 1),
        "p99_sec": round(percentile(secs, 99), 1),
        "min_sec": round(secs[0], 1) if secs else 0.0,
        "max_sec": round(secs[-1], 1) if secs else 0.0,
        "mean_sec": round(sum(secs) / len(secs), 1) if secs else 0.0,
        "by_status": by_status,
    }
