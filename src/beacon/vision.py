from pathlib import Path

import structlog

from beacon import claude

log = structlog.get_logger()

SYSTEM_PROMPT = """You are a satellite imagery analyst specializing in disaster verification.

The user will give you paths to multiple satellite products for a claim:
- BEFORE / AFTER Sentinel-2 True Color (TCI) tiles (~10m/pixel)
- Optionally a dNBR (delta Normalized Burn Ratio) heatmap, computed from B08 (NIR)
  and B12 (SWIR2). Colors: RED = burned/severely changed (positive dNBR),
  GREEN = unchanged or regrowth (negative dNBR), YELLOW = transitional. dNBR is the
  diagnostic signal for forest burn scars that TCI alone can miss.
- Optionally a Sentinel-1 SAR change map (radar log-ratio in dB). RED = backscatter
  decrease (typical signature of floods/water inundation, sometimes burn-cleared
  ground); GREEN = backscatter increase (new corner reflectors → buildings, vehicles).
  SAR sees through clouds and works at night, so it is the primary evidence when TCI
  is cloud-occluded.

Use the Read tool to load each available image, then output a single JSON object —
no prose, no markdown fences, just JSON — with these exact keys:

{
  "verdict": "supported" | "refuted" | "inconclusive",
  "confidence": 0.0,
  "evidence": ["specific visual observation 1", "specific visual observation 2"],
  "contradictions": ["observation that argues against the claim"],
  "geospatial_delta_estimate": "qualitative description of change between BEFORE and AFTER, citing TCI and dNBR if both available",
  "cloud_or_quality_issues": "any concerns about image clarity, cloud cover, or coverage"
}

Be concrete. Reference image regions ("northwest quadrant", "along the ridgeline").
When evidence is weak, say so plainly — "inconclusive" is a valid verdict. When dNBR
is available and shows large red regions consistent with the claimed bbox, that is
strong evidence even if the TCI looks similar before/after."""


def _build_prompt(
    claim_text: str,
    place: str,
    event_date: str,
    before_path: Path | None,
    after_path: Path | None,
    dnbr_path: Path | None,
    dnbr_burn_pct: float | None,
    s1_change_path: Path | None,
    s1_decrease_pct: float | None,
) -> str:
    if before_path and after_path:
        availability_note = ""
    elif before_path:
        availability_note = (
            "\n\nNote: AFTER image is unavailable (no clear-sky Sentinel-2 pass after the event "
            "date yet). Reason about whether the location matches the claim, but unless the "
            "BEFORE image already shows clear evidence, mark verdict=inconclusive."
        )
    elif after_path:
        availability_note = "\n\nNote: BEFORE image is unavailable."
    else:
        availability_note = "\n\nNote: NO imagery is available."

    before_str = str(before_path) if before_path else "(unavailable)"
    after_str = str(after_path) if after_path else "(unavailable)"
    dnbr_section = ""
    if dnbr_path:
        pct_note = f" (burn coverage above dNBR > 0.27: {dnbr_burn_pct:.1f}% of pixels)" if dnbr_burn_pct is not None else ""
        dnbr_section = f"\ndNBR change map: {dnbr_path}{pct_note}"

    s1_section = ""
    if s1_change_path:
        s1_pct_note = f" (backscatter decrease > 3 dB: {s1_decrease_pct:.1f}% of pixels)" if s1_decrease_pct is not None else ""
        s1_section = f"\nSentinel-1 SAR change map: {s1_change_path}{s1_pct_note}"

    return (
        f"Analyze these satellite products for evidence of the claim.\n\n"
        f"CLAIM: {claim_text[:1500]}\n"
        f"LOCATION: {place or 'unknown'}\n"
        f"EVENT DATE (approx): {event_date or 'unknown'}\n\n"
        f"BEFORE TCI: {before_str}\n"
        f"AFTER TCI:  {after_str}"
        f"{dnbr_section}"
        f"{s1_section}\n\n"
        f"Read every available image using the Read tool, then output the JSON verdict."
        f"{availability_note}"
    )


def analyze_tile_pair(
    *,
    claim_text: str,
    place: str,
    event_date: str,
    before_path: Path | None,
    after_path: Path | None,
    dnbr_path: Path | None = None,
    dnbr_burn_pct: float | None = None,
    s1_change_path: Path | None = None,
    s1_decrease_pct: float | None = None,
    cwd: str | None = None,
) -> dict:
    if before_path is None and after_path is None and dnbr_path is None and s1_change_path is None:
        return {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence": [],
            "contradictions": [],
            "geospatial_delta_estimate": "no imagery available",
            "cloud_or_quality_issues": "no tiles fetched",
        }
    prompt = _build_prompt(
        claim_text, place, event_date, before_path, after_path,
        dnbr_path, dnbr_burn_pct, s1_change_path, s1_decrease_pct,
    )
    raw = claude.ask(
        prompt,
        system_prompt=SYSTEM_PROMPT,
        max_turns=4,
        allowed_tools=["Read"],
        cwd=cwd,
        permission_mode="bypassPermissions",
    )
    parsed = claude.parse_json_block(raw)
    if not parsed:
        log.warning("vision.unparseable", raw_preview=raw[:300])
        return {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence": [],
            "contradictions": [],
            "geospatial_delta_estimate": "parse_error",
            "cloud_or_quality_issues": "model output could not be parsed as JSON",
            "_raw": raw[:1000],
        }
    return parsed
