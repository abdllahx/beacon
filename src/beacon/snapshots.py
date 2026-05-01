"""Versioned eval snapshots — turn 'are we moving in the right direction' from a
vibe-check into a measurable thing.

Each snapshot captures:
  - timestamp + git SHA + dirty flag (so we know which code produced this)
  - active tunable values (so threshold sweeps are auditable)
  - full eval-report metrics output (the headline numbers)
  - per-event verdicts + centroid distances + IoU (so we can find regressions)

Snapshots are JSON files under `data/eval_snapshots/{timestamp}_{label}.json` —
no DB table, so they're trivially git-versionable / diffable / blog-attachable.
"""

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import structlog

from beacon import db, eval_metrics, tunables

log = structlog.get_logger()

SNAPSHOT_DIR = Path("data/eval_snapshots")


def _git_sha() -> tuple[str, bool]:
    """Return (sha, is_dirty). ('uncommitted', True) if no git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted", True


def _capture_tunables() -> dict:
    """Snapshot every UPPERCASE constant in beacon.tunables for reproducibility."""
    return {
        name: getattr(tunables, name)
        for name in dir(tunables)
        if name.isupper() and not name.startswith("_")
    }


def _per_event_rows() -> list[dict]:
    """One row per completed benchmark run, with predicted vs ground-truth distance + iou."""
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                br.id,
                ee.dis_no,
                ee.disaster_type,
                ee.country,
                dgt.gt_source,
                br.beacon_verdict,
                br.beacon_confidence,
                ST_X(dgt.gt_centroid) AS gt_lon,
                ST_Y(dgt.gt_centroid) AS gt_lat,
                ST_X(ST_Centroid(c.bbox)) AS pred_lon,
                ST_Y(ST_Centroid(c.bbox)) AS pred_lat,
                CASE
                    WHEN ST_Area(ST_Union(c.bbox, dgt.gt_bbox)) > 0
                    THEN ST_Area(ST_Intersection(c.bbox, dgt.gt_bbox)) / ST_Area(ST_Union(c.bbox, dgt.gt_bbox))
                    ELSE NULL
                END AS iou
            FROM benchmark_runs br
            JOIN emdat_events ee ON ee.id = br.emdat_event_id
            JOIN disaster_ground_truth dgt ON dgt.emdat_event_id = ee.id
            JOIN claims c ON c.id = br.claim_id
            WHERE br.beacon_run_id IS NOT NULL
            ORDER BY br.id
            """
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    out = []
    for r in rows:
        distance_km = None
        if all(r.get(k) is not None for k in ("gt_lon", "gt_lat", "pred_lon", "pred_lat")):
            distance_km = round(
                eval_metrics.haversine_km(r["gt_lat"], r["gt_lon"], r["pred_lat"], r["pred_lon"]),
                1,
            )
        out.append(
            {
                "benchmark_run_id": r["id"],
                "dis_no": r["dis_no"],
                "disaster_type": r["disaster_type"],
                "country": r["country"],
                "gt_source": r["gt_source"],
                "verdict": r["beacon_verdict"],
                "confidence": float(r["beacon_confidence"]) if r["beacon_confidence"] is not None else None,
                "centroid_distance_km": distance_km,
                "iou": round(float(r["iou"]), 4) if r["iou"] is not None else None,
            }
        )
    return out


def capture(label: str, *, gdis_only: bool = False) -> Path:
    """Capture a snapshot to disk. Returns the snapshot file path."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "snapshot"
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    sha, dirty = _git_sha()
    metrics = eval_metrics.compute_metrics(gdis_only=gdis_only)
    snapshot = {
        "id": f"{ts}_{safe_label}",
        "label": label,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "gdis_only": gdis_only,
        "tunables": _capture_tunables(),
        "metrics": metrics,
        "per_event": _per_event_rows(),
    }
    out_path = SNAPSHOT_DIR / f"{ts}_{safe_label}.json"
    out_path.write_text(json.dumps(snapshot, indent=2, default=str))
    log.info("snapshot.captured", path=str(out_path), n_events=len(snapshot["per_event"]))
    return out_path


def load(snapshot_id: str) -> dict:
    """Load a snapshot by id (the {timestamp}_{label} prefix or full filename)."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    matches = list(SNAPSHOT_DIR.glob(f"*{snapshot_id}*.json"))
    if not matches:
        raise FileNotFoundError(f"no snapshot matching {snapshot_id!r} in {SNAPSHOT_DIR}")
    if len(matches) > 1:
        raise ValueError(f"snapshot id {snapshot_id!r} ambiguous; matched {[m.name for m in matches]}")
    return json.loads(matches[0].read_text())


def list_snapshots() -> list[dict]:
    if not SNAPSHOT_DIR.exists():
        return []
    out = []
    for f in sorted(SNAPSHOT_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            out.append(
                {
                    "id": d.get("id"),
                    "label": d.get("label"),
                    "created_at": d.get("created_at"),
                    "git_sha": d.get("git_sha"),
                    "git_dirty": d.get("git_dirty"),
                    "n": (d.get("metrics") or {}).get("n"),
                }
            )
        except (ValueError, KeyError):
            continue
    return out


def _flatten_metrics(m: dict, prefix: str = "") -> dict[str, float | int | str | None]:
    """Flatten the nested metrics dict to dotted keys for diffing."""
    out: dict = {}
    if not isinstance(m, dict):
        return out
    for k, v in m.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, prefix=full))
        else:
            out[full] = v
    return out


def diff(snapshot_a_id: str, snapshot_b_id: str) -> dict:
    """Compute deltas: snapshot_b - snapshot_a, plus per-event regressions."""
    a = load(snapshot_a_id)
    b = load(snapshot_b_id)

    a_flat = _flatten_metrics(a.get("metrics") or {})
    b_flat = _flatten_metrics(b.get("metrics") or {})
    keys = sorted(set(a_flat) | set(b_flat))
    metric_deltas: dict[str, dict] = {}
    for k in keys:
        av, bv = a_flat.get(k), b_flat.get(k)
        if isinstance(av, int | float) and isinstance(bv, int | float):
            metric_deltas[k] = {"a": av, "b": bv, "delta": round(bv - av, 4)}
        else:
            metric_deltas[k] = {"a": av, "b": bv, "delta": None}

    # Per-event regressions: events present in both snapshots whose verdict changed
    a_by_dis = {e["dis_no"]: e for e in a.get("per_event") or []}
    b_by_dis = {e["dis_no"]: e for e in b.get("per_event") or []}
    shared = sorted(set(a_by_dis) & set(b_by_dis))
    verdict_changes: list[dict] = []
    distance_regressions: list[dict] = []
    for d in shared:
        ea, eb = a_by_dis[d], b_by_dis[d]
        if ea.get("verdict") != eb.get("verdict"):
            verdict_changes.append(
                {
                    "dis_no": d,
                    "type": eb.get("disaster_type"),
                    "from": ea.get("verdict"),
                    "to": eb.get("verdict"),
                    "from_conf": ea.get("confidence"),
                    "to_conf": eb.get("confidence"),
                }
            )
        a_d, b_d = ea.get("centroid_distance_km"), eb.get("centroid_distance_km")
        if isinstance(a_d, int | float) and isinstance(b_d, int | float) and abs(b_d - a_d) > 5:
            distance_regressions.append(
                {
                    "dis_no": d,
                    "type": eb.get("disaster_type"),
                    "from_km": a_d,
                    "to_km": b_d,
                    "delta_km": round(b_d - a_d, 1),
                }
            )

    return {
        "a": {"id": a.get("id"), "n": (a.get("metrics") or {}).get("n")},
        "b": {"id": b.get("id"), "n": (b.get("metrics") or {}).get("n")},
        "metric_deltas": metric_deltas,
        "verdict_changes": verdict_changes,
        "distance_regressions": distance_regressions,
        "events_only_in_a": sorted(set(a_by_dis) - set(b_by_dis)),
        "events_only_in_b": sorted(set(b_by_dis) - set(a_by_dis)),
    }
