"""Aggregate metrics over completed benchmark runs.

Ground truth comes from `disaster_ground_truth`: GDIS (peer-reviewed, Rosvold & Buhaug
2021) when available, EM-DAT's own coordinates otherwise. The eval report stratifies
by gt_source so we can cite "GDIS-validated subset" for the resume-grade numbers and
keep the broader self-geocoded subset as a secondary indicator.

Geoparsing accuracy follows the literature convention (Gritta et al. 2018, "A pragmatic
guide to geoparsing evaluation"): report Accuracy@N where N is a distance threshold in
km. Standard reporting: @10km, @50km, @161km.
"""

import math

from beacon import db
from beacon.tunables import ACCURACY_THRESHOLDS_KM


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _aggregate(rows: list[tuple], label: str) -> dict:
    if not rows:
        return {"n": 0, "label": label}
    verdicts: dict[str, int] = {}
    confs: list[float] = []
    distances: list[float] = []
    ious: list[float] = []
    by_type: dict[str, dict] = {}
    by_source: dict[str, int] = {}
    for verdict, conf, dtype, src, gt_lon, gt_lat, p_lon, p_lat, iou, _dnbr, _sar in rows:
        v = verdict or "null"
        verdicts[v] = verdicts.get(v, 0) + 1
        if conf is not None:
            confs.append(float(conf))
        if all(x is not None for x in (gt_lon, gt_lat, p_lon, p_lat)):
            distances.append(haversine_km(gt_lat, gt_lon, p_lat, p_lon))
        if iou is not None:
            ious.append(float(iou))
        by_type.setdefault(dtype or "unknown", {"n": 0, "supported": 0})
        by_type[dtype or "unknown"]["n"] += 1
        if verdict == "supported":
            by_type[dtype or "unknown"]["supported"] += 1
        by_source[src or "unknown"] = by_source.get(src or "unknown", 0) + 1

    accuracy_at: dict[str, float | None] = {}
    for thresh in ACCURACY_THRESHOLDS_KM:
        if distances:
            accuracy_at[f"@{thresh}km"] = round(
                sum(1 for d in distances if d <= thresh) / len(distances), 3
            )
        else:
            accuracy_at[f"@{thresh}km"] = None

    distances_sorted = sorted(distances)

    n = len(rows)
    return {
        "label": label,
        "n": n,
        "ground_truth_source": dict(by_source),
        "verdicts": dict(verdicts),
        "recall_supported": round(verdicts.get("supported", 0) / n, 3),
        "inconclusive_rate": round(verdicts.get("inconclusive", 0) / n, 3),
        "refuted_rate": round(verdicts.get("refuted", 0) / n, 3),
        "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
        "geoparsing": {
            "n_with_distance": len(distances),
            "mean_centroid_distance_km": round(sum(distances) / len(distances), 1) if distances else None,
            "median_centroid_distance_km": round(distances_sorted[len(distances_sorted) // 2], 1) if distances_sorted else None,
            "accuracy": accuracy_at,
        },
        "spatial": {
            "n_with_iou": len(ious),
            "mean_bbox_iou": round(sum(ious) / len(ious), 3) if ious else None,
        },
        "by_type": {
            t: {
                "n": d["n"],
                "recall_supported": round(d["supported"] / d["n"], 3),
            }
            for t, d in by_type.items()
        },
    }


def _fetch_rows(gdis_only: bool):
    where_extra = "AND dgt.gt_source = 'gdis'" if gdis_only else ""
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                br.beacon_verdict,
                br.beacon_confidence,
                ee.disaster_type,
                dgt.gt_source,
                ST_X(dgt.gt_centroid) AS gt_lon,
                ST_Y(dgt.gt_centroid) AS gt_lat,
                ST_X(ST_Centroid(c.bbox)) AS pred_lon,
                ST_Y(ST_Centroid(c.bbox)) AS pred_lat,
                CASE
                    WHEN ST_Area(ST_Union(c.bbox, dgt.gt_bbox)) > 0
                    THEN ST_Area(ST_Intersection(c.bbox, dgt.gt_bbox)) / ST_Area(ST_Union(c.bbox, dgt.gt_bbox))
                    ELSE NULL
                END AS iou,
                NULLIF(v.imagery_metadata->'nbr'->'delta'->>'burn_pct', '')::float AS dnbr_burn_pct,
                NULLIF(v.imagery_metadata->'s1'->'change'->>'decrease_pct', '')::float AS s1_decrease_pct
            FROM benchmark_runs br
            JOIN emdat_events ee ON ee.id = br.emdat_event_id
            JOIN disaster_ground_truth dgt ON dgt.emdat_event_id = ee.id
            JOIN claims c ON c.id = br.claim_id
            LEFT JOIN verification_runs v ON v.id = br.beacon_run_id
            WHERE br.beacon_run_id IS NOT NULL
              {where_extra}
            """
        )
        return cur.fetchall()


def compute_metrics(*, gdis_only: bool = False) -> dict:
    """Compute metrics. By default reports both the GDIS-validated subset (rigorous)
    and the full set (broader, includes self-geocoded ground truth)."""
    if gdis_only:
        return _aggregate(_fetch_rows(gdis_only=True), label="gdis_validated_only")
    full_rows = _fetch_rows(gdis_only=False)
    gdis_rows = [r for r in full_rows if r[3] == "gdis"]
    return {
        "headline_gdis_validated": _aggregate(gdis_rows, label="gdis_validated_only"),
        "broader_self_geocoded": _aggregate(full_rows, label="full_set_including_emdat_self"),
    }
