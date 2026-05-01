"""Cost log helpers.

Estimates Claude Sonnet 4.5 cost from char counts when the Agent SDK doesn't
surface token counts. Pricing as of 2026-04 (Anthropic public list):
    Sonnet 4.5: $3 / 1M input tokens, $15 / 1M output tokens
A rough 4 chars-per-token heuristic gives a stable lower bound for dashboards
without per-call instrumentation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from beacon import db


SONNET_INPUT_USD_PER_TOK = 3.0 / 1_000_000
SONNET_OUTPUT_USD_PER_TOK = 15.0 / 1_000_000
CHARS_PER_TOK = 4.0


@dataclass(slots=True)
class CostRow:
    operation: str
    input_chars: int = 0
    output_chars: int = 0
    latency_ms: int | None = None
    model: str = "claude-sonnet-4-5"
    provider: str = "claude"
    run_id: int | None = None


def estimate_usd(input_chars: int, output_chars: int) -> float:
    in_tok = input_chars / CHARS_PER_TOK
    out_tok = output_chars / CHARS_PER_TOK
    return round(in_tok * SONNET_INPUT_USD_PER_TOK + out_tok * SONNET_OUTPUT_USD_PER_TOK, 6)


def log_event(row: CostRow) -> None:
    in_tok = int(row.input_chars / CHARS_PER_TOK) if row.input_chars else None
    out_tok = int(row.output_chars / CHARS_PER_TOK) if row.output_chars else None
    cost = estimate_usd(row.input_chars, row.output_chars)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cost_events
              (run_id, provider, model, operation, input_tokens, output_tokens, cost_usd, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (row.run_id, row.provider, row.model, row.operation, in_tok, out_tok, cost, row.latency_ms),
        )
        conn.commit()


def aggregate() -> dict:
    """Return per-operation cost + total + run-level summary for dashboard."""
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT operation,
                   COUNT(*)            AS calls,
                   SUM(cost_usd)       AS cost_usd,
                   AVG(latency_ms)     AS mean_latency_ms,
                   SUM(input_tokens)   AS in_tok,
                   SUM(output_tokens)  AS out_tok
            FROM cost_events
            GROUP BY operation
            ORDER BY cost_usd DESC NULLS LAST
            """
        )
        per_op = [
            {
                "operation": r[0],
                "calls": int(r[1]),
                "cost_usd": float(r[2] or 0),
                "mean_latency_ms": float(r[3] or 0),
                "in_tok": int(r[4] or 0),
                "out_tok": int(r[5] or 0),
            }
            for r in cur.fetchall()
        ]
        cur.execute("SELECT COUNT(*), SUM(cost_usd) FROM cost_events")
        n, total = cur.fetchone()
        cur.execute(
            """
            SELECT run_id, SUM(cost_usd) AS run_cost
            FROM cost_events
            WHERE run_id IS NOT NULL
            GROUP BY run_id
            """
        )
        per_run = [float(r[1] or 0) for r in cur.fetchall()]
    per_run.sort()
    median_run = per_run[len(per_run) // 2] if per_run else 0.0
    return {
        "total_calls": int(n or 0),
        "total_cost_usd": float(total or 0),
        "median_run_cost_usd": median_run,
        "per_operation": per_op,
        "n_runs_with_cost": len(per_run),
    }


def aggregate_json() -> str:
    return json.dumps(aggregate(), indent=2)


def backfill_from_runs() -> dict:
    """Estimate Claude cost for past verification_runs that don't have cost_events.

    The Agent SDK doesn't surface token counts retroactively, so we approximate:
    each run consumed (1) one synthesize call (system prompt + inputs JSON +
    final_report_md) and (2) one vision_vqa call (system prompt + ~600 chars
    imagery summary + ~1500-char vision_verdict JSON).

    Marks rows with operation='synthesize_estimated' / 'vision_vqa_estimated' so
    they are clearly distinguishable from live-logged events.
    """
    SYNTH_SYS_PROMPT_CHARS = 600
    SYNTH_INPUTS_CHARS = 1500
    VISION_SYS_PROMPT_CHARS = 700
    VISION_INPUTS_CHARS = 600
    VISION_OUTPUT_CHARS = 1500

    inserted = 0
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.id, COALESCE(LENGTH(v.final_report_md), 0) AS report_chars
            FROM verification_runs v
            LEFT JOIN cost_events c ON c.run_id = v.id
            WHERE v.status = 'synth_done'
              AND c.id IS NULL
            """
        )
        rows = cur.fetchall()
        for run_id, report_chars in rows:
            log_event(CostRow(
                operation="synthesize_estimated",
                input_chars=SYNTH_SYS_PROMPT_CHARS + SYNTH_INPUTS_CHARS,
                output_chars=int(report_chars or 1500),
                run_id=run_id,
            ))
            log_event(CostRow(
                operation="vision_vqa_estimated",
                input_chars=VISION_SYS_PROMPT_CHARS + VISION_INPUTS_CHARS,
                output_chars=VISION_OUTPUT_CHARS,
                run_id=run_id,
            ))
            inserted += 2
    return {"runs_backfilled": len(rows), "events_inserted": inserted}
