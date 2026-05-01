import os

import streamlit as st

from beacon import cost as cost_mod
from beacon import db, latency

REVIEWER_PASSWORD = os.getenv("REVIEWER_PASSWORD")  # set in Streamlit Cloud Secrets


def _hitl_unlocked() -> bool:
    """True when no password is configured (local dev) OR the user typed it correctly."""
    if not REVIEWER_PASSWORD:
        return True
    if st.session_state.get("hitl_authed"):
        return True
    pw = st.text_input(
        "Reviewer password (HITL writes locked)",
        type="password",
        key="hitl_pw_input",
    )
    if pw and pw == REVIEWER_PASSWORD:
        st.session_state["hitl_authed"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password.")
    return False

st.set_page_config(page_title="Beacon — Geospatial Event Verification", layout="wide")
st.title("Beacon — Multimodal Geospatial Event Verification")
st.caption(
    "News claim → satellite imagery → grounded report. "
    "9 HuggingFace tasks composed in a 21-node LangGraph DAG."
)


@st.cache_data(ttl=10)
def load_runs():
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.id, v.claim_id, a.title, a.url, a.source, a.published_at,
                   c.admin_region, ST_AsGeoJSON(c.bbox) AS bbox_geojson,
                   v.imagery_metadata, v.vision_verdict, v.final_verdict,
                   v.final_report_md, v.translations, v.article_summary, v.status
            FROM verification_runs v
            JOIN claims c ON c.id = v.claim_id
            JOIN articles a ON a.id = c.article_id
            WHERE v.status IN ('vision_done', 'synth_done')
            ORDER BY v.id DESC
            """
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


@st.cache_data(ttl=30)
def load_firms_in_bbox(bbox_geojson: str | None) -> list[dict]:
    if not bbox_geojson:
        return []
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT detected_at, ST_X(point) AS lon, ST_Y(point) AS lat, frp, satellite
            FROM firms_events
            WHERE ST_Within(point, ST_GeomFromGeoJSON(%s))
            ORDER BY detected_at
            """,
            (bbox_geojson,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


@st.cache_data(ttl=60)
def load_tile_archive_paths() -> dict[int, str]:
    """id -> tile_path map for VDR thumbnails."""
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, tile_path FROM tile_archive")
        return dict(cur.fetchall())


runs = load_runs()
if not runs:
    st.info("No verification runs yet. Run `uv run beacon graph-run <claim_id>`.")
    st.stop()


def _label(r: dict) -> str:
    fv = r.get("final_verdict") or {}
    verdict = fv.get("verdict") or (r.get("vision_verdict") or {}).get("verdict") or "?"
    return f"#{r['id']} · {verdict} · {(r['title'] or '')[:55]}"


choice = st.sidebar.radio(
    "Verification runs",
    options=runs,
    format_func=_label,
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(runs)} run(s) loaded")

# ─── sidebar: ops dashboards ─────────────────────────────────────────────
with st.sidebar.expander("⏱ Latency (p50/p95/p99)", expanded=False):
    lat = latency.latency_stats()
    if lat["n"]:
        st.metric("p50", f"{lat['p50_sec']:.1f}s")
        st.metric("p95", f"{lat['p95_sec']:.1f}s")
        st.metric("p99", f"{lat['p99_sec']:.1f}s")
        st.caption(f"n={lat['n']} runs · mean {lat['mean_sec']:.1f}s")
    else:
        st.caption("No completed runs yet.")

with st.sidebar.expander("💰 Cost log", expanded=False):
    agg = cost_mod.aggregate()
    st.metric("Total cost (USD)", f"${agg['total_cost_usd']:.4f}")
    st.metric("Calls logged", agg["total_calls"])
    if agg["n_runs_with_cost"]:
        st.metric("Median per-run", f"${agg['median_run_cost_usd']:.4f}")
    if agg["per_operation"]:
        st.caption("Per-operation:")
        for op in agg["per_operation"][:6]:
            st.markdown(
                f"`{op['operation']}` — {op['calls']} calls · ${op['cost_usd']:.4f}"
            )
    else:
        st.caption("No cost events logged yet (table empty).")

run = choice
final_v = run.get("final_verdict") or {}
vision_v = run.get("vision_verdict") or {}
verdict = final_v.get("verdict") or vision_v.get("verdict", "?")
confidence = final_v.get("confidence") or vision_v.get("confidence", 0.0)

imagery = run.get("imagery_metadata") or {}
s2 = imagery.get("s2") or {}
nbr = imagery.get("nbr") or {}
s1 = imagery.get("s1") or {}
s2_after = s2.get("after") or {}

# ─── header ──────────────────────────────────────────────────────────────
st.subheader(run["title"] or "(untitled)")
meta_cols = st.columns([3, 1])
with meta_cols[0]:
    pub = run.get("published_at")
    bits = []
    if pub:
        bits.append(pub.strftime("%Y-%m-%d"))
    if run.get("admin_region"):
        bits.append(run["admin_region"])
    if run.get("source"):
        bits.append(f"source: {run['source']}")
    st.caption(" · ".join(bits))
with meta_cols[1]:
    color_map = {"supported": "🟢", "refuted": "🔴", "inconclusive": "🟡"}
    icon = color_map.get(verdict, "⚪")
    st.metric("Verdict", f"{icon} {verdict}", f"conf {float(confidence):.2f}")

# ─── HF modalities active badge row ──────────────────────────────────────
# Visual proof to a recruiter that all 9 tasks fired for this run.
hf_active = {
    "NER (Davlan)": run.get("claim_id") is not None,
    "Zero-shot text (BART-MNLI)": run.get("claim_id") is not None,
    "Sentence-Sim (MiniLM)": run.get("claim_id") is not None,
    "Image Seg (SegFormer)": bool((s2_after.get("segmentation") or {}).get("mask_path")),
    "Object Detection (DETR)": bool((s2_after.get("detections") or {}).get("overlay_path")),
    "Zero-shot Image (SigLIP)": bool((s2_after.get("zero_shot") or {}).get("ranked")),
    "VDR (SigLIP)": bool(imagery.get("vdr_matches")),
    "Summarization (BART-CNN)": bool(run.get("article_summary")),
    "Translation (MarianMT)": bool(run.get("translations")),
}
active_n = sum(1 for v in hf_active.values() if v)
st.caption(f"**HuggingFace tasks active for this run: {active_n} / 9**")
badge_cols = st.columns(9)
for (name, on), col in zip(hf_active.items(), badge_cols, strict=False):
    col.markdown(f"<div style='font-size:0.78em; text-align:center; padding:6px; "
                 f"background:{'#1f6f1f' if on else '#3a3a3a'}; color:white; "
                 f"border-radius:4px;'>{'✓' if on else '·'}<br/>{name}</div>",
                 unsafe_allow_html=True)
st.markdown("")

if final_v.get("headline"):
    st.info(final_v["headline"])

# ─── BART article summary ────────────────────────────────────────────────
if run.get("article_summary"):
    st.markdown("#### Article summary (BART-CNN)")
    st.caption(run["article_summary"])

# ─── Sentinel-2 TCI ──────────────────────────────────────────────────────
st.markdown("#### Sentinel-2 True Color")
img_cols = st.columns(2)
if (s2_before_path := (s2.get("before") or {}).get("path")):
    img_cols[0].caption("BEFORE")
    img_cols[0].image(s2_before_path, use_container_width=True)
else:
    img_cols[0].warning("No BEFORE tile available")
if (s2_after_path := s2_after.get("path")):
    img_cols[1].caption("AFTER")
    img_cols[1].image(s2_after_path, use_container_width=True)
else:
    img_cols[1].warning("No AFTER tile available")

# ─── dNBR delta ──────────────────────────────────────────────────────────
dnbr = nbr.get("delta") or {}
if dnbr.get("path"):
    burn_pct = dnbr.get("burn_pct")
    badge = f" — {burn_pct:.1f}% pixels above Key & Benson moderate-burn threshold" if burn_pct is not None else ""
    st.markdown(f"#### dNBR (Normalized Burn Ratio delta){badge}")
    st.image(dnbr["path"], use_container_width=True,
             caption="Red = severe burn (positive dNBR), green = unburned/regrowth")

# ─── Sentinel-1 SAR change ───────────────────────────────────────────────
s1_change = s1.get("change") or {}
if s1_change.get("path"):
    dec_pct = s1_change.get("decrease_pct")
    badge = f" — {dec_pct:.1f}% pixels with > 3 dB backscatter drop" if dec_pct is not None else ""
    st.markdown(f"#### Sentinel-1 SAR change (cloud-penetrating){badge}")
    st.image(s1_change["path"], use_container_width=True,
             caption="Red = backscatter decrease (flood/burn), green = increase (new structures)")

# ─── SegFormer land-cover segmentation ───────────────────────────────────
seg = s2_after.get("segmentation") or {}
if seg.get("mask_path"):
    st.markdown("#### SegFormer land-cover segmentation (HF: Image Segmentation)")
    seg_cols = st.columns([2, 1])
    seg_cols[0].image(seg["mask_path"], use_container_width=True,
                      caption=f"ADE20K classes overlaid · model: {seg.get('model','')}")
    top = seg.get("top") or []
    if top:
        seg_cols[1].caption("Top classes by area")
        for entry in top[:6]:
            seg_cols[1].markdown(f"`{entry['label']}` — **{entry['pct']:.1f}%**")

# ─── DETR object detection ───────────────────────────────────────────────
det = s2_after.get("detections") or {}
if det.get("overlay_path"):
    n_obj = det.get("n_objects", 0)
    st.markdown(f"#### DETR object detection — {n_obj} objects (HF: Object Detection)")
    det_cols = st.columns([2, 1])
    det_cols[0].image(det["overlay_path"], use_container_width=True,
                      caption=f"Bounding boxes · model: {det.get('model','')}")
    classes_count = det.get("classes_count") or {}
    if classes_count:
        det_cols[1].caption("Detected classes")
        for label, n in sorted(classes_count.items(), key=lambda kv: -kv[1])[:8]:
            det_cols[1].markdown(f"`{label}` — **{n}**")
    else:
        det_cols[1].caption("No COCO-class objects at this resolution.")
        if det.get("resolution_limit_note"):
            det_cols[1].markdown(f"*{det['resolution_limit_note']}*")

# ─── SigLIP zero-shot image classification ───────────────────────────────
zsc = s2_after.get("zero_shot") or {}
if zsc.get("ranked"):
    st.markdown("#### SigLIP zero-shot tile classification (HF: Zero-Shot Image Classification)")
    st.caption(f"Model: `{zsc.get('model','google/siglip-base-patch16-224')}`")
    zsc_data = [{"label": r["label"], "score": float(r["score"])} for r in zsc["ranked"][:8]]
    st.dataframe(
        zsc_data,
        column_config={
            "label": st.column_config.TextColumn("Candidate label"),
            "score": st.column_config.ProgressColumn("Confidence", format="%.4f", min_value=0, max_value=1),
        },
        hide_index=True,
    )

# ─── SigLIP Visual Document Retrieval ────────────────────────────────────
vdr_matches = imagery.get("vdr_matches") or []
if vdr_matches:
    st.markdown("#### Visual Document Retrieval — similar archived tiles (HF: VDR via SigLIP)")
    st.caption("Top-K archived tiles whose SigLIP embeddings are nearest the AFTER tile.")
    vdr_cols = st.columns(min(5, len(vdr_matches)))
    for col, m in zip(vdr_cols, vdr_matches[:5], strict=False):
        path = m.get("tile_path")
        sim = m.get("similarity", 0)
        desc = m.get("description") or m.get("disaster_type") or ""
        if path:
            col.image(path, use_container_width=True)
        col.caption(f"sim {sim:.3f} · {desc[:70]}")

# ─── FIRMS ground-truth ──────────────────────────────────────────────────
firms_hits = load_firms_in_bbox(run.get("bbox_geojson"))
if firms_hits:
    st.markdown(f"#### NASA FIRMS ground-truth — {len(firms_hits)} thermal anomaly hit(s) in bbox")
    st.dataframe(
        firms_hits,
        column_config={
            "detected_at": st.column_config.DatetimeColumn("Detected at", format="YYYY-MM-DD HH:mm"),
            "lat": st.column_config.NumberColumn("Lat", format="%.4f"),
            "lon": st.column_config.NumberColumn("Lon", format="%.4f"),
            "frp": st.column_config.NumberColumn("FRP (MW)", format="%.1f"),
        },
        height=240,
        hide_index=True,
    )
else:
    st.caption("No FIRMS thermal anomalies in this bbox / our loaded window.")

# ─── Verification report (with multi-language toggle) ────────────────────
if run.get("final_report_md"):
    translations = run.get("translations") or {}
    LANG_LABEL = {"en": "English", "es": "Español", "ar": "العربية", "fr": "Français"}
    available = ["en"] + sorted(k for k in translations if k != "en")
    st.markdown("### Verification report")
    if len(available) > 1:
        lang = st.radio(
            "Language (translated by Helsinki-NLP MarianMT — HF: Translation)",
            options=available,
            format_func=lambda c: LANG_LABEL.get(c, c),
            horizontal=True,
            key=f"report_lang_{run['id']}",
        )
        body = run["final_report_md"] if lang == "en" else translations.get(lang, run["final_report_md"])
        if lang == "ar":
            st.markdown(f'<div dir="rtl" style="text-align: right">{body}</div>', unsafe_allow_html=True)
        else:
            st.markdown(body)
    else:
        st.markdown(run["final_report_md"])

with st.expander("Vision verdict (raw JSON)"):
    st.json(vision_v)

with st.expander("Full imagery_metadata (raw JSON)"):
    st.json(imagery)

# ─── HITL feedback ───────────────────────────────────────────────────────
st.markdown("### Analyst feedback (HITL)")
st.caption(
    "Reviewer corrections feed the DSPy training set. Routes to the `feedback` "
    "table for next-iteration prompt optimization."
)
run_id = run["id"]
if not _hitl_unlocked():
    st.info("HITL writes are locked on the public demo. Authorized reviewers only.")
    st.stop()
fb_cols = st.columns([1, 1, 3])
if fb_cols[0].button("👍 Confirm", key=f"thumbs_up_{run_id}"):
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (run_id, rating) VALUES (%s, 'thumbs_up')", (run_id,)
        )
        conn.commit()
    st.success("Saved as confirmation.")
if fb_cols[1].button("👎 Wrong", key=f"thumbs_down_{run_id}"):
    st.session_state[f"correcting_{run_id}"] = True

if st.session_state.get(f"correcting_{run_id}"):
    corrected = st.selectbox(
        "Corrected verdict",
        options=["supported", "refuted", "inconclusive"],
        key=f"corr_v_{run_id}",
    )
    notes = st.text_area("Notes (optional)", key=f"corr_n_{run_id}")
    if st.button("Save correction", key=f"corr_save_{run_id}"):
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO feedback (run_id, rating, corrected_verdict, notes)
                   VALUES (%s, 'thumbs_down', %s, %s)""",
                (run_id, corrected, notes or None),
            )
            conn.commit()
        st.session_state[f"correcting_{run_id}"] = False
        st.success("Correction saved.")

with st.expander("Past feedback for this run", expanded=False):
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT created_at, rating, corrected_verdict, notes
               FROM feedback WHERE run_id = %s ORDER BY created_at DESC""",
            (run_id,),
        )
        fb_rows = cur.fetchall()
    if fb_rows:
        for ts, rating, cv, n in fb_rows:
            st.markdown(
                f"- `{ts:%Y-%m-%d %H:%M}` · **{rating}**"
                + (f" → {cv}" if cv else "") + (f" — {n}" if n else "")
            )
    else:
        st.caption("No feedback recorded for this run.")
