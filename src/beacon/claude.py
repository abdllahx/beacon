import asyncio
import json
import re
import time

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

from beacon import cost
from beacon.observability import observe

log = structlog.get_logger()


DEFAULT_MODEL = "claude-sonnet-4-5"  # pin Sonnet to preserve Max pool (Opus = ~5× cost)


async def _ask_async(
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_turns: int = 1,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
    permission_mode: str | None = None,
    model: str | None = DEFAULT_MODEL,
) -> str:
    parts: list[str] = []
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=max_turns,
        allowed_tools=allowed_tools or [],
        cwd=cwd,
        permission_mode=permission_mode,
        model=model,
    )
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
    return "".join(parts).strip()


@observe(name="claude.ask", as_type="generation")
def ask(
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_turns: int = 1,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
    permission_mode: str | None = None,
    model: str | None = DEFAULT_MODEL,
    operation: str = "ask",
    run_id: int | None = None,
) -> str:
    """Synchronous one-shot Claude call via the Agent SDK (Max subscription).

    Logs a cost_events row tagged with `operation` for the cost dashboard.
    Wrapped in @observe so calls trace to Langfuse when configured.
    """
    started = time.time()
    result = asyncio.run(
        _ask_async(
            prompt,
            system_prompt=system_prompt,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
            cwd=cwd,
            permission_mode=permission_mode,
            model=model,
        )
    )
    elapsed_ms = int((time.time() - started) * 1000)
    try:
        cost.log_event(cost.CostRow(
            operation=operation,
            input_chars=len(prompt) + len(system_prompt or ""),
            output_chars=len(result),
            latency_ms=elapsed_ms,
            model=model or DEFAULT_MODEL,
            run_id=run_id,
        ))
    except Exception as e:
        log.warning("cost.log_failed", err=str(e))
    return result


def parse_json_block(text: str) -> dict | None:
    """Extract the first JSON object from text, tolerating ```json fences."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
