import os
import sqlite3
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from beacon.config import get_settings
from beacon.graph import nodes
from beacon.graph.state import BeaconState

CHECKPOINT_DB = Path("data/checkpoints.sqlite")


def _wire_langsmith() -> None:
    """If LANGSMITH_TRACING is enabled in config, propagate the env vars LangGraph
    looks for so every node call auto-streams to https://smith.langchain.com."""
    settings = get_settings()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        # langsmith library reads either LANGSMITH_* or LANGCHAIN_*
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint


_wire_langsmith()


def build_graph(checkpointer=None):
    """Compile the Beacon verification DAG.

    Phase A topology (current):

        START
          │
          ▼
        load_claim ── init_run
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
        fetch_s2_before     fetch_s2_after
                └─────────┬─────────┘
                          ▼
                      vision_vqa
                          │
                          ▼
                    persist_vision
                          │
                          ▼
                      synthesize
                          │
                          ▼
                         END

    Phase B will add fetch_s1_*, render_nbr, segment, detect as parallel siblings of
    the s2 fetches, all feeding vision_vqa. Phase C adds translate after synthesize.
    """
    g: StateGraph = StateGraph(BeaconState)

    g.add_node("extract_claim", nodes.extract_claim)
    g.add_node("geocode_claim", nodes.geocode_claim_node)
    g.add_node("load_claim", nodes.load_claim)
    g.add_node("init_run", nodes.init_run)
    g.add_node("fetch_s2_before", nodes.fetch_s2_before)
    g.add_node("fetch_s2_after", nodes.fetch_s2_after)
    g.add_node("fetch_nbr_before", nodes.fetch_nbr_before)
    g.add_node("fetch_nbr_after", nodes.fetch_nbr_after)
    g.add_node("compute_dnbr", nodes.compute_dnbr)
    g.add_node("fetch_s1_before", nodes.fetch_s1_before)
    g.add_node("fetch_s1_after", nodes.fetch_s1_after)
    g.add_node("compute_s1_change", nodes.compute_s1_change)
    g.add_node("segment_after", nodes.segment_after)
    g.add_node("detect_after", nodes.detect_after)
    g.add_node("classify_tile", nodes.classify_tile)
    g.add_node("vdr_search", nodes.vdr_search)
    g.add_node("vision_vqa", nodes.vision_vqa)
    g.add_node("persist_vision", nodes.persist_vision)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("summarize_article", nodes.summarize_article)
    g.add_node("translate_report", nodes.translate_report)

    g.add_edge(START, "extract_claim")
    g.add_edge("extract_claim", "geocode_claim")
    g.add_edge("geocode_claim", "load_claim")
    g.add_edge("load_claim", "init_run")
    # 6-way parallel imagery fetch: TCI before/after + NBR before/after + S1 SAR before/after
    for parallel in (
        "fetch_s2_before",
        "fetch_s2_after",
        "fetch_nbr_before",
        "fetch_nbr_after",
        "fetch_s1_before",
        "fetch_s1_after",
    ):
        g.add_edge("init_run", parallel)
    # dNBR / s1_change depend on their two parents
    g.add_edge("fetch_nbr_before", "compute_dnbr")
    g.add_edge("fetch_nbr_after", "compute_dnbr")
    g.add_edge("fetch_s1_before", "compute_s1_change")
    g.add_edge("fetch_s1_after", "compute_s1_change")
    # SegFormer land-cover segmentation runs after the AFTER S2 tile is available.
    g.add_edge("fetch_s2_after", "segment_after")
    g.add_edge("fetch_s2_after", "detect_after")
    # SigLIP zero-shot classification + VDR retrieval — both consume the AFTER tile.
    g.add_edge("fetch_s2_after", "classify_tile")
    g.add_edge("fetch_s2_after", "vdr_search")
    # vision_vqa fans in from TCI + dNBR + S1 change
    g.add_edge("fetch_s2_before", "vision_vqa")
    g.add_edge("fetch_s2_after", "vision_vqa")
    g.add_edge("compute_dnbr", "vision_vqa")
    g.add_edge("compute_s1_change", "vision_vqa")
    g.add_edge("segment_after", "vision_vqa")
    g.add_edge("detect_after", "vision_vqa")
    g.add_edge("classify_tile", "vision_vqa")
    g.add_edge("vdr_search", "vision_vqa")
    g.add_edge("vision_vqa", "persist_vision")
    # synthesize and summarize_article run in parallel after persist_vision —
    # synthesize uses Claude to write the long-form verdict, summarize_article uses
    # BART-CNN to extractive-summarize the source article (HF Summarization task).
    g.add_edge("persist_vision", "synthesize")
    g.add_edge("persist_vision", "summarize_article")
    g.add_edge("synthesize", "translate_report")
    g.add_edge("summarize_article", "translate_report")
    g.add_edge("translate_report", END)

    return g.compile(checkpointer=checkpointer)


def run_pipeline(
    claim_id: int | None = None,
    *,
    article_id: int | None = None,
    thread_id: str | None = None,
) -> dict:
    """Execute the DAG with SQLite-backed checkpointing.

    Pass `article_id` to run the FULL pipeline (extract → geocode → imagery → vision → synth)
    from raw article text — used by the v2 benchmark to test geocoder accuracy honestly.

    Pass `claim_id` to skip extract + geocode (the demo path; the claim is already
    populated with bbox by hand or by upstream CLI commands).

    Pass the same `thread_id` to resume a previous run from its last successful node.
    """
    if claim_id is None and article_id is None:
        raise ValueError("run_pipeline requires either claim_id or article_id")

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    graph = build_graph(checkpointer=saver)
    seed = claim_id if claim_id is not None else f"art-{article_id}"
    tid = thread_id or f"{seed}-{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": tid}}

    initial: BeaconState = {"errors": []}
    if claim_id is not None:
        initial["claim_id"] = claim_id
    if article_id is not None:
        initial["article_id"] = article_id

    final_state = graph.invoke(initial, config=config)
    final_state["_thread_id"] = tid
    return final_state


def render_mermaid() -> str:
    """Return the DAG as a Mermaid diagram string (no checkpointer needed)."""
    graph = build_graph(checkpointer=None)
    return graph.get_graph().draw_mermaid()
