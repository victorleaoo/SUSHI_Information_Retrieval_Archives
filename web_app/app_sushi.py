import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import utils_experiments_viz as u1
import utils_topics_viz as u2
import utils_retrieval as u3
import utils_ecf_inspector as u4
try:
    from streamlit_scroll_to_top import scroll_to_here
except ImportError:
    scroll_to_here = None

st.set_page_config(layout="wide", page_title="SUSHI BAR")

# ============================================================
# DYNAMIC THEME & CSS SYSTEM
# ============================================================

# ============================================================
# SHARED CSS (ALWAYS LIGHT MODE)
# ============================================================
st.markdown("""
<style>
    /* App Shell & Background */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #334155 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:last-child {
        border-top: 1px solid #E2E8F0 !important;
        margin-top: 12px;
        padding-top: 12px;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #0F172A !important;
    }
    p, span, label, div[data-testid="stMarkdownContainer"] p, .stMarkdown {
        color: #334155 !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.85rem !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
        word-break: break-word !important;
    }
    /* Hide Streamlit metric delta arrows */
    div[data-testid="stMetricDelta"] svg {
        display: none !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9 !important;
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        color: #475569 !important;
        padding: 8px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E40AF, #1D4ED8) !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] span, .stTabs [aria-selected="true"] p {
        color: #FFFFFF !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }

    /* Custom Boxes & Cards */
    .info-box {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin: 8px 0 !important;
        color: #1E40AF !important;
    }
    .info-box strong { color: #1E3A8A !important; }

    .folder-card {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 14px !important;
        margin-bottom: 10px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }
    .folder-card.rel3 { border-left: 4px solid #16A34A !important; }
    .folder-card.rel1 { border-left: 4px solid #D97706 !important; }
    .folder-card.rel0 { border-left: 4px solid #94A3B8 !important; }

    .metric-box {
        background-color: #F8FAFC !important;
        border-radius: 8px !important;
        padding: 12px !important;
        text-align: center !important;
        border: 1px solid #E2E8F0 !important;
    }
    .kpi-lg { font-size: 24px; font-weight: bold; }
    .kpi-md { font-size: 20px; font-weight: bold; }
    .kpi-sm { font-size: 14px; font-weight: bold; }
    .kpi-label { color: #64748B !important; font-size: 13px; margin-top: 4px; }
    .kpi-xs { font-size: 11px; color: #64748B !important; }
    .kpi-good { color: #16A34A !important; }
    .kpi-bad { color: #DC2626 !important; }
    .kpi-mid { color: #D97706 !important; }
    .kpi-accent { color: #1D4ED8 !important; }
    .grade-3 { color: #16A34A !important; font-size: 22px; font-weight: bold; }
    .grade-1 { color: #D97706 !important; font-size: 22px; font-weight: bold; }
    .grade-0 { color: #94A3B8 !important; font-size: 22px; font-weight: bold; }

    .tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px; }
    .tag-top5 { background-color: #DBEAFE !important; color: #1E40AF !important; }
    .tag-top15 { background-color: #F1F5F9 !important; color: #475569 !important; }
    .tag-movable { background-color: #DCFCE7 !important; color: #15803D !important; }
</style>
""", unsafe_allow_html=True)


def _apply_light_theme(fig, height=500):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))
    fig.update_yaxes(gridcolor="#E2E8F0", title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))
    return fig


# ============================================================
# DATA OVERVIEW HELPERS
# ============================================================
_BASE_DIR_DA = Path(__file__).resolve().parent.parent
_FOLDERS_PATH_DA = _BASE_DIR_DA / "data" / "folders_metadata" / "FoldersV1.3.json"
_ITEMS_PATH_DA = _BASE_DIR_DA / "data" / "items_metadata" / "itemsV1.2.json"

_DA_COLORS = {
    "primary": "#1D4ED8",     # Cobalt Blue
    "secondary": "#EA580C",   # Dark Orange / Rust
    "accent": "#047857",      # Forest / Emerald Green
    "success": "#15803D",     # Deep Green
    "warning": "#D97706",     # Dark Amber
    "danger": "#DC2626",      # Red
    "info": "#2563EB",        # Cobalt Blue
    "rich": "#047857",        # Emerald Green
    "moderate": "#EA580C",    # Dark Orange
    "poor": "#DC2626",        # Red
}

_STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from",
    "as","is","was","are","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","this","that",
    "these","those","it","its","he","she","they","them","his","her","their","we","our",
    "my","your","not","no","nor","also","than","other","which","who","whom","what",
    "when","where","how","all","each","every","both","few","more","most","some","any",
    "such","only","own","same","so","very","just","about","up","out","into","over",
    "after","before","between","under","above","below","during","through","while",
    "against","documents","document","including","new","united","states","government",
    "brazil","brazilian","u.s.","u.s","state","department","embassy","president",
    "regarding","report","according","one","two","three","four","five","among","well",
    "however","several","various","includes","related","discussed","described",
    "reported","mentioned","involved","provided","based","within",
    "1964","1965","1966","1967","1968","1969","1970","1971","1972","1973","1974",
}


def _apply_light_theme(fig, height=500):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))
    fig.update_yaxes(gridcolor="#E2E8F0", title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))
    return fig


def _extract_keywords(texts, top_n=25):
    wc = Counter()
    for text in texts:
        if not text or not isinstance(text, str): continue
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
        wc.update(w for w in words if w not in _STOP_WORDS)
    return wc.most_common(top_n)


@st.cache_data(show_spinner="Loading folder metadata…")
def _load_folders_df():
    with open(_FOLDERS_PATH_DA, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [{"folder_id": fid, **meta} for fid, meta in data.items()]
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["end_date"] = pd.to_datetime(df["endDate"], format="%m/%d/%Y", errors="coerce")
    df["snc_clean"] = df["snc"].fillna("").astype(str).str.strip()
    df["snc_primary"] = df["snc_clean"].apply(lambda x: x.split()[0] if x else "")
    def _snc2(s):
        parts = s.split()
        if len(parts) < 2: return s
        return f"{parts[0]} {parts[1].split('-')[0]}"
    df["snc_2level"] = df["snc_clean"].apply(_snc2)
    df["snc_3level"] = df["snc_clean"]
    df["has_scope"] = df["raw_scope"].fillna("").str.strip().apply(lambda x: bool(x) and x.lower() != "nan")
    df["scope_text"] = df["raw_scope"].fillna("").str.strip()
    df["start_year"] = df["start_date"].dt.year
    return df


@st.cache_data(show_spinner="Loading document metadata…")
def _load_items_df():
    with open(_ITEMS_PATH_DA, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for doc_id, meta in data.items():
        ocr_pages = meta.get("ocr", [])
        ocr_text = " ".join(ocr_pages) if isinstance(ocr_pages, list) else ""
        rows.append({
            "doc_id": doc_id,
            "box": meta.get("Sushi Box", ""),
            "folder_id": meta.get("Sushi Folder", ""),
            "title": meta.get("title", "") or meta.get("Brown Title", ""),
            "date": meta.get("date", ""),
            "summary": meta.get("summary", ""),
            "ocr_pages": len(ocr_pages) if isinstance(ocr_pages, list) else 0,
            "ocr_length": len(ocr_text),
            "has_summary": bool(meta.get("summary", "")),
            "has_title": bool(meta.get("title", "") or meta.get("Brown Title", "")),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Building folder-document index…")
def _build_folder_doc_index(_items_df):
    folder_docs = _items_df.groupby("folder_id")["doc_id"].apply(list).to_dict()
    folder_doc_counts = _items_df.groupby("folder_id")["doc_id"].count().to_dict()
    return folder_docs, folder_doc_counts


# ============================================================
# EXPERIMENT ANALYZER UI COMPONENTS
# ============================================================

def _get_sorted_topic_labels(df_chart, sort_mode="topic_order"):
    """Return sorted topic labels based on the chosen mode."""
    title_map = u1.get_topic_title_map()
    if sort_mode == "ndcg_desc":
        # Sort by mean nDCG descending
        topic_means = df_chart.groupby('Topic')['nDCG'].mean().sort_values(ascending=False)
        return [title_map.get(t, t) for t in topic_means.index]
    else:
        # Default: topic order T1, T2, ...
        return [title_map.get(f'T{i}', f'T{i}') for i in range(1, 46)]


def render_charts(df_chart: pd.DataFrame, topics_to_display: list, sort_mode: str = "topic_order"):
    if df_chart.empty or not topics_to_display: return
    title_map = u1.get_topic_title_map()

    if sort_mode == "ndcg_desc":
        topic_means = df_chart.groupby('Topic')['nDCG'].mean().sort_values(ascending=False)
        sorted_topics = [t for t in topic_means.index if t in topics_to_display]
        topic_labels = [title_map.get(t, t) for t in sorted_topics]
    else:
        topic_labels = [title_map.get(t, t) for t in topics_to_display]

    chart_data = df_chart[df_chart['Topic'].isin(topics_to_display)].copy()
    if 'Topic Label' not in chart_data.columns:
        chart_data['Topic Label'] = chart_data['Topic'].map(lambda x: title_map.get(x, x))
    present_types = list(chart_data['Type'].unique())
    domain = [t for t in present_types if t in u1.COLOR_MAP] + [t for t in present_types if t not in u1.COLOR_MAP]
    range_colors = [u1.get_model_color(t, i) for i, t in enumerate(domain)]
    color_scale = alt.Color('Type', scale=alt.Scale(domain=domain, range=range_colors),
                            legend=alt.Legend(title="Model Type", orient="top", columns=3, labelLimit=2000, titleLimit=2000))
    base = alt.Chart(chart_data).encode(
        y=alt.Y('Topic Label:N', title="Topics", sort=topic_labels, axis=alt.Axis(labelLimit=1000))
    )
    rule_bg = base.mark_line(color='lightgray', strokeDash=[2,2], opacity=0.3).encode(
        x='min(nDCG)', x2='max(nDCG)', detail='Topic Label'
    )
    ci_rule = base.mark_rule(opacity=0.6, thickness=2).encode(x='min_ci', x2='max_ci', color=color_scale)
    tick_min = base.mark_tick(thickness=2, height=12).encode(x='min_ci', color=color_scale)
    tick_max = base.mark_tick(thickness=2, height=12).encode(x='max_ci', color=color_scale)
    points = base.mark_circle(size=120, opacity=1).encode(
        x=alt.X('nDCG', title="Mean nDCG@5 with 95% CI"),
        color=color_scale,
        tooltip=['Topic Label', 'Type', 'nDCG', 'min_ci', 'max_ci']
    )
    final_chart = (rule_bg + ci_rule + tick_min + tick_max + points).properties(height=len(topics_to_display) * 65)
    st.altair_chart(final_chart, width="stretch")


def render_two_run_comparison_chart(df_run_a, df_run_b, run_a_name, run_b_name, sort_mode="topic_order"):
    if df_run_a.empty and df_run_b.empty:
        st.warning("No metric data available to render comparison chart.")
        return
    title_map = u1.get_topic_title_map()

    df_a = df_run_a.copy(); df_b = df_run_b.copy()
    label_a = f"Run A: {run_a_name}"; label_b = f"Run B: {run_b_name}"
    df_a['Run'] = label_a; df_b['Run'] = label_b
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    if 'Topic Label' not in df_combined.columns:
        df_combined['Topic Label'] = df_combined['Topic'].map(lambda x: title_map.get(x, x))

    if sort_mode == "ndcg_desc":
        topic_means = df_combined.groupby('Topic')['nDCG'].mean().sort_values(ascending=False)
        sorted_labels = [title_map.get(t, t) for t in topic_means.index]
    else:
        sorted_labels = [title_map.get(f'T{i}', f'T{i}') for i in range(1, 46)]

    color_scale = alt.Color('Run:N',
        scale=alt.Scale(domain=[label_a, label_b], range=['#1D4ED8', '#EA580C']),
        legend=alt.Legend(title="Experiment Run", orient="top"))
    base = alt.Chart(df_combined).encode(
        y=alt.Y('Topic Label:N', title="Topics & Titles", sort=sorted_labels, axis=alt.Axis(labelLimit=1000))
    )
    rule_bg = base.mark_line(color='lightgray', strokeDash=[2,2], opacity=0.3).encode(
        x='min_ci:Q', x2='max_ci:Q', detail='Topic Label:N'
    )
    ci_rule = base.mark_rule(opacity=0.7, thickness=2.5).encode(x='min_ci:Q', x2='max_ci:Q', color=color_scale)
    tick_min = base.mark_tick(thickness=2, height=10).encode(x='min_ci:Q', color=color_scale)
    tick_max = base.mark_tick(thickness=2, height=10).encode(x='max_ci:Q', color=color_scale)
    points = base.mark_circle(size=120, opacity=0.85).encode(
        x=alt.X('nDCG:Q', title="Mean nDCG@5 with 95% CI", scale=alt.Scale(domain=[0, 1])),
        color=color_scale,
        tooltip=['Topic Label', 'Run', 'nDCG', 'min_ci', 'max_ci', 'Relevance']
    )
    chart = (rule_bg + ci_rule + tick_min + tick_max + points).properties(
        title=f"Topic Performance Comparison: {run_a_name} (Blue) vs {run_b_name} (Orange)",
        height=len(sorted_labels) * 28
    )
    st.altair_chart(chart, width="stretch")


def render_seed_variance_chart(var_df: pd.DataFrame):
    """Render a horizontal box plot showing per-topic nDCG@5 distributions across seeds."""
    if var_df.empty:
        return
    import plotly.graph_objects as go

    x_vals = []
    y_vals = []
    for _, row in var_df.iloc[::-1].iterrows():
        title = row["Topic Title"]
        for val in row["Values"]:
            x_vals.append(val)
            y_vals.append(title)

    fig = go.Figure()
    fig.add_trace(go.Box(
        x=x_vals,
        y=y_vals,
        orientation="h",
        name="",
        marker_color="#1D4ED8",
        boxpoints=False,
        fillcolor="rgba(29, 78, 216, 0.2)",
        line=dict(color="#1E40AF", width=1.5),
        hoveron="boxes",
        xhoverformat=".3f",
        hovertemplate=(
            "<b>Median:</b> %{median:.3f}<br>"
            "<b>Q1:</b> %{q1:.3f} &nbsp; <b>Q3:</b> %{q3:.3f}<br>"
            "<b>Min:</b> %{min:.3f} &nbsp; <b>Max:</b> %{max:.3f}"
            "<extra></extra>"
        ),
    ))

    fig = _apply_light_theme(fig, max(500, len(var_df) * 36))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="nDCG@5", range=[-0.02, 1.05], hoverformat=".3f"),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=10, r=20, t=10, b=30),
    )
    st.plotly_chart(fig, width="stretch")


def render_retrieval_analysis(run_name: str, run_dir: str, all_topics: dict, folders_meta: dict, top_n: int = 15):
    """Render the Retrieval Analysis section: top-N retrieved folders per topic."""
    if not run_dir or not os.path.exists(run_dir):
        return

    run_txt_path = os.path.join(run_dir, "run.txt")
    if not os.path.isfile(run_txt_path):
        return

    folder_qrels = u3.load_folder_qrels()
    layer23 = u3.load_layer23_for_run(run_dir)

    # Summary table for all topics
    st.subheader("📋 All Topics Summary")
    summary_df = u3.compute_all_topics_summary(run_dir, folder_qrels, top_n_values=[5, 10, 15, 20])
    if not summary_df.empty:
        st.dataframe(summary_df, width="stretch", hide_index=True, height=400)
    else:
        st.info("No ranking data found for this run.")
        return

    # Per-topic detail
    st.subheader(f"📊 Top-{top_n} Retrieved Folders")
    topic_rankings = u3.load_run_rankings(run_dir, top_n=top_n)

    topic_ids_sorted = sorted(all_topics.keys())
    topic_options = [f"{tid} — {all_topics[tid].get('TITLE', '')}" for tid in topic_ids_sorted]

    sel_topic_str = st.selectbox(
        "Select Topic for Retrieval Analysis:",
        options=topic_options,
        key="retrieval_topic_select"
    )
    sel_topic_id = sel_topic_str.split(" — ")[0]
    topic_data = all_topics[sel_topic_id]
    topic_qrels = folder_qrels.get(sel_topic_id, {})
    ranking = topic_rankings.get(sel_topic_id, [])

    # Topic summary metrics
    ndcg5 = u3.ndcg_at_k(ranking, topic_qrels, k=5)
    n_rel_top5 = sum(1 for _, fid in ranking[:5] if topic_qrels.get(fid, 0) > 0)
    n_rel_topn = sum(1 for _, fid in ranking[:top_n] if topic_qrels.get(fid, 0) > 0)
    outside_top5 = [fid for _, fid in ranking[5:top_n] if topic_qrels.get(fid, 0) > 0]
    has_potential = len(outside_top5) > 0

    col_t1, col_t2, col_t3, col_t4 = st.columns([3, 1, 1, 1])
    with col_t1:
        st.markdown(f"**{sel_topic_id}: {topic_data.get('TITLE', '')}**")
        st.caption(topic_data.get('DESCRIPTION', ''))
    with col_t2:
        ndcg_cls = "kpi-good" if ndcg5 > 0.4 else "kpi-bad" if ndcg5 < 0.2 else "kpi-mid"
        st.markdown(f'<div class="metric-box"><div class="kpi-lg {ndcg_cls}">{ndcg5:.4f}</div><div class="kpi-label">nDCG@5</div></div>', unsafe_allow_html=True)
    with col_t3:
        st.markdown(f'<div class="metric-box"><div class="kpi-md kpi-accent">{n_rel_top5}/5</div><div class="kpi-label">Rel in Top-5</div></div>', unsafe_allow_html=True)
    with col_t4:
        p_cls = "kpi-good" if has_potential else "kpi-bad"
        p_label = "✅ Rerank Potential" if has_potential else "❌ No Potential"
        st.markdown(f'<div class="metric-box"><div class="kpi-sm {p_cls}">{p_label}</div><div class="kpi-xs">{len(outside_top5)} rel at rank 6-{top_n}</div></div>', unsafe_allow_html=True)

    if not ranking:
        st.info("No ranking data for this topic in the selected run.")
        return

    for rank, folder_id in ranking:
        grade = topic_qrels.get(folder_id, 0)
        folder_data = folders_meta.get(folder_id, {})
        aug_data = layer23.get(folder_id, {})

        in_top5 = rank <= 5
        is_movable = (not in_top5) and grade > 0

        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        top_tag = '<span class="tag tag-top5">TOP-5</span>' if in_top5 else f'<span class="tag tag-top15">rank 6-{top_n}</span>'
        # Use star-based grade display for consistency with Topic Viewer
        grade_label = u3.grade_stars(grade)
        movable_tag = '<span class="tag tag-movable">🎯 MOVABLE</span>' if is_movable else ''

        label = folder_data.get('label', 'N/A')
        snc = folder_data.get('snc', 'N/A')
        parent_exp = folder_data.get('label_parent_expanded', '')

        with st.expander(f"{rank_emoji} Rank {rank} — {folder_id} — {label} — {grade_label}", expanded=False):
            col_f1, col_f2 = st.columns([3, 1])
            with col_f1:
                st.markdown(f"**SNC:** `{snc}` | **Box:** `{folder_data.get('box', 'N/A')}` | **Date:** `{folder_data.get('date', 'N/A')}`")
                if parent_exp:
                    st.markdown(f"**Parent Classification:** {parent_exp}")
                if folder_data.get('raw_scope'):
                    st.markdown(f"**Scope Note:** {str(folder_data['raw_scope'])[:300]}…")
                st.markdown(top_tag + movable_tag, unsafe_allow_html=True)
            with col_f2:
                grade_cls_map = {3: "grade-3", 1: "grade-1", 0: "grade-0"}
                g_cls = grade_cls_map.get(grade, "grade-0")
                st.markdown(f'<div class="metric-box"><div class="{g_cls}">Grade {grade}</div></div>', unsafe_allow_html=True)
            if aug_data:
                with st.expander("🤖 LLM Augmentation"):
                    if aug_data.get('CORE_THEMES'): st.markdown(f"**CORE_THEMES:** {aug_data['CORE_THEMES']}")
                    if aug_data.get('RELATED_CONCEPTS'): st.markdown(f"**RELATED_CONCEPTS:** {aug_data['RELATED_CONCEPTS']}")



def run_single_experiment_ui():
    """Screen 1: Single Experiment Analysis & Retrieval Analysis"""
    st.title("🔬 Single Experiment Viewer")
    st.caption("Select an experiment folder to see per-model nDCG@5 results, seed variance, and detailed retrieval rankings.")

    all_runs_df = u1.get_all_runs_statistics()
    if all_runs_df.empty:
        st.error("No valid run folders found.")
        return

    grouped_runs = u1.get_grouped_run_configurations()
    all_run_names = u1.sort_run_names(all_runs_df['Run Name'].tolist())

    selected_run = st.selectbox(
        "Select Experiment:",
        options=all_run_names,
        index=0 if all_run_names else None,
        help="Select an experiment run folder from all_runs/."
    )

    if selected_run:
        df_chart, _, all_topics, model_results = u1.process_experiment_data([selected_run], {selected_run: [selected_run]})
        st.subheader("Global Performance (Mean nDCG@5)")
        for model_key, m_info in model_results.items():
            stats = m_info['stats']
            count = m_info['count']
            st.metric(label=f"{selected_run}", value=f"{stats['val']:.4f} ± {stats['margin']:.3f}")

        if all_topics:
            with st.expander("📈 Topic Performance Chart (nDCG@5)", expanded=True):
                render_charts(df_chart, all_topics, sort_mode="topic_order")

            with st.expander("📊 Seed Variance per Topic", expanded=True):
                var_df = u1.compute_seed_variance_df(selected_run)
                if var_df.empty:
                    st.info("No multi-seed metric data available for this model run.")
                elif var_df["N_Seeds"].max() <= 1:
                    st.caption(f"Only 1 seed available for `{selected_run}` — no cross-seed variance to display.")
                else:
                    n_seeds = int(var_df["N_Seeds"].max())
                    st.caption(f"Distribution of nDCG@5 across **{n_seeds} seeds** for `{selected_run}`.")
                    render_seed_variance_chart(var_df)
        else:
            st.warning("No topic data found for this experiment.")

    # ── Retrieval Analysis ──
    if selected_run:
        r_dir = u1.resolve_run_folder_path(selected_run) or ""
        if r_dir and os.path.isfile(os.path.join(r_dir, "run.txt")):
            st.markdown("---")
            st.header("🔬 Retrieval Analysis")
            st.caption("Inspect the retrieved folders per topic alongside qrels grades for the selected run.")

            top_n_sel = st.selectbox("Top-N retrieved folders to show:", [5, 10, 15, 20], index=2, key="ret_top_n")

            folders_meta, _ = u2.load_metadata()
            ecf_data = u2.load_ecf_data()
            all_ecf_topics = {}
            if "ExperimentSets" in ecf_data:
                for es in ecf_data["ExperimentSets"]:
                    if "Topics" in es: all_ecf_topics.update(es["Topics"])

            if all_ecf_topics:
                render_retrieval_analysis(selected_run, r_dir, all_ecf_topics, folders_meta, top_n=top_n_sel)


def run_two_experiment_ui():
    """Screen 2: Direct Two-Experiment Overlay Comparison"""
    st.title("⚔️ Two-Experiment Viewer")
    st.caption("Directly compare two experiments side-by-side with statistical significance tests and topic separation analysis.")

    all_runs_df = u1.get_all_runs_statistics()
    if all_runs_df.empty:
        st.warning("No runs found for comparison.")
        return

    all_run_names = u1.sort_run_names(all_runs_df['Run Name'].tolist())

    col_sel_a, col_sel_b = st.columns(2)
    with col_sel_a:
        st.subheader("🟦 Experiment A Selection")
        sel_run_a = st.selectbox(
            "Select Experiment A (Blue):",
            options=all_run_names,
            index=0 if all_run_names else 0,
            key="side_by_side_run_a"
        )

    with col_sel_b:
        st.subheader("🟧 Experiment B Selection")
        sel_run_b = st.selectbox(
            "Select Experiment B (Orange):",
            options=all_run_names,
            index=1 if len(all_run_names) > 1 else 0,
            key="side_by_side_run_b"
        )

    if sel_run_a and sel_run_b:
        folder_a = u1.resolve_run_folder_path(sel_run_a) or ""
        stats_a = u1.load_overall_stats(folder_a)
        rel_a = u1.calculate_global_relevance_mean(u1.load_relevance_stats(folder_a))
        mean_a, margin_a = stats_a.get('mean', 0.0), stats_a.get('margin', 0.0)

        folder_b = u1.resolve_run_folder_path(sel_run_b) or ""
        stats_b = u1.load_overall_stats(folder_b)
        rel_b = u1.calculate_global_relevance_mean(u1.load_relevance_stats(folder_b))
        mean_b, margin_b = stats_b.get('mean', 0.0), stats_b.get('margin', 0.0)

        delta_ndcg = mean_a - mean_b
        m_col_a, m_col_b, m_col_delta = st.columns([2, 2, 1])
        with m_col_a:
            st.metric(label=f"🟦 Run A: {sel_run_a}", value=f"{mean_a:.4f} ± {margin_a:.4f}", help=f"Global Relevance: {rel_a:.2f}")
        with m_col_b:
            st.metric(label=f"🟧 Run B: {sel_run_b}", value=f"{mean_b:.4f} ± {margin_b:.4f}", help=f"Global Relevance: {rel_b:.2f}")
        with m_col_delta:
            st.metric(label="Global Delta (A - B)", value=f"{delta_ndcg:+.4f}")

        # Wilcoxon signed-rank test
        wilcoxon_result = u1.run_wilcoxon_test(sel_run_a, sel_run_b)
        if "error" not in wilcoxon_result:
            p_val = wilcoxon_result["p_value"]
            sig_icon = "✅" if wilcoxon_result["significant"] else "❌"
            sig_text = "Significant" if wilcoxon_result["significant"] else "Not Significant"
            raw_winner = wilcoxon_result["winner"]
            if raw_winner in ("Run A", sel_run_a):
                winner_disp = "🟦 Run A"
            elif raw_winner in ("Run B", sel_run_b):
                winner_disp = "🟧 Run B"
            else:
                winner_disp = raw_winner

            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                st.metric(
                    label="Wilcoxon p-value",
                    value=f"{p_val:.5f}",
                    delta=f"{sig_icon} {sig_text} (α=0.05)",
                    delta_color="off"
                )
            with w_col2:
                st.metric(
                    label="Winner",
                    value=winner_disp,
                    delta=f"Experiment A: {wilcoxon_result['wins_a']} wins | Experiment B: {wilcoxon_result['wins_b']} wins",
                    delta_color="off"
                )
            with w_col3:
                st.metric(
                    label="Seeds Compared",
                    value=f"{wilcoxon_result['n_seeds']} seeds",
                    delta=f"A wins: {wilcoxon_result['wins_a']} | B wins: {wilcoxon_result['wins_b']}",
                    delta_color="off"
                )
        elif wilcoxon_result.get("error"):
            st.caption(f"⚠️ Wilcoxon test: {wilcoxon_result['error']}")

        df_run_a = u1.get_single_run_topic_chart_dataset(sel_run_a)
        df_run_b = u1.get_single_run_topic_chart_dataset(sel_run_b)

        with st.expander("📊 Topic Performance Overlay Chart", expanded=True):
            render_two_run_comparison_chart(df_run_a, df_run_b, sel_run_a, sel_run_b, sort_mode="topic_order")

        with st.expander("🎯 Topic Separation Analysis (Experiment A vs Experiment B)", expanded=True):
            st.caption("Categorization of topics by relative mean nDCG difference: **Better** (≥ +10%), **About Equal** (between -10% and +10%), and **Worse** (≤ -10%).")
            categories = u1.categorize_topics_comparison(df_run_a, df_run_b)
            df_better = categories['better']; df_equal = categories['equal']; df_worse = categories['worse']
            total_topics = len(df_better) + len(df_equal) + len(df_worse)
            col_b, col_e, col_w = st.columns(3)
            with col_b:
                pct_b = (len(df_better) / total_topics * 100) if total_topics else 0.0
                st.metric("🟢 Experiment A Better than B (≥ +10%)", f"{len(df_better)} topics ({pct_b:.1f}%)")
            with col_e:
                pct_e = (len(df_equal) / total_topics * 100) if total_topics else 0.0
                st.metric("🟡 Experiment A About Equal to B (within ±10%)", f"{len(df_equal)} topics ({pct_e:.1f}%)")
            with col_w:
                pct_w = (len(df_worse) / total_topics * 100) if total_topics else 0.0
                st.metric("🔴 Experiment A Worse than B (≤ -10%)", f"{len(df_worse)} topics ({pct_w:.1f}%)")

            tab_better, tab_equal, tab_worse = st.tabs([
                f"🟢 Experiment A Better ({len(df_better)})",
                f"🟡 Experiment A About Equal ({len(df_equal)})",
                f"🔴 Experiment A Worse ({len(df_worse)})"
            ])

            def _fmt_sep(df_cat):
                if df_cat.empty: return pd.DataFrame()
                res = df_cat.copy()
                res['Experiment A nDCG'] = res['nDCG_A'].apply(lambda x: f"{x:.4f}")
                res['Experiment B nDCG'] = res['nDCG_B'].apply(lambda x: f"{x:.4f}")
                res['Delta (A - B)'] = res['Diff'].apply(lambda x: f"{x:+.4f}")
                res['% Difference'] = res['Pct_Diff'].apply(lambda x: f"{x * 100:+.2f}%")
                return res[['Topic Label', 'Experiment A nDCG', 'Experiment B nDCG', 'Delta (A - B)', '% Difference']]

            with tab_better:
                if not df_better.empty: st.dataframe(_fmt_sep(df_better), width="stretch", hide_index=True)
                else: st.info("No topics where Run A is at least 10% better than Run B.")
            with tab_equal:
                if not df_equal.empty: st.dataframe(_fmt_sep(df_equal), width="stretch", hide_index=True)
                else: st.info("No topics where Run A is within ±10% of Run B.")
            with tab_worse:
                if not df_worse.empty: st.dataframe(_fmt_sep(df_worse), width="stretch", hide_index=True)
                else: st.info("No topics where Run A is at least 10% worse than Run B.")


# ============================================================
# DATA OVERVIEW UI
# ============================================================

def run_data_overview_ui():
    st.title("📦 Collection Viewer — SUSHI Collection")

    folders_df = _load_folders_df()
    items_df = _load_items_df()
    _, folder_doc_counts = _build_folder_doc_index(items_df)
    folders_df["doc_count"] = folders_df["folder_id"].map(folder_doc_counts).fillna(0).astype(int)

    # ── SECTION 1: Overview Metrics ──────────────────────────────────
    st.markdown("## 📊 Collection Statistics")
    n_folders = len(folders_df); n_docs = len(items_df)
    n_boxes = folders_df["box"].nunique(); n_snc_distinct = folders_df["snc_3level"].nunique()
    n_snc_primary = folders_df["snc_primary"].nunique()

    c3, c1, c2 = st.columns(3)
    c3.metric("📦 Total Boxes", f"{n_boxes:,}"); c1.metric("📁 Total Folders", f"{n_folders:,}"); c2.metric("📄 Total Documents", f"{n_docs:,}")
    c4, c5 = st.columns(2)
    c4.metric("🏷️ Distinct 3-level SNCs", f"{n_snc_distinct}"); c5.metric("🏷️ 1-Level SNC Codes", f"{n_snc_primary}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Documents per Folder")
        fig = px.histogram(folders_df, x="doc_count", nbins=50, color_discrete_sequence=[_DA_COLORS["primary"]],
            labels={"doc_count": "Documents", "count": "Folders"})
        st.plotly_chart(_apply_light_theme(fig, 350), width="stretch")
    with col_b:
        st.markdown("### Folders per Box")
        fpb = folders_df.groupby("box").size().reset_index(name="folder_count")
        fig = px.histogram(fpb, x="folder_count", nbins=30, color_discrete_sequence=[_DA_COLORS["primary"]],
            labels={"folder_count": "Folders", "count": "Boxes"})
        st.plotly_chart(_apply_light_theme(fig, 350), width="stretch")

    st.markdown("### 1-Level SNC Code — Folder & Document Counts")
    primary_stats = folders_df.groupby("snc_primary").agg(
        folder_count=("folder_id","count"), total_docs=("doc_count","sum")
    ).reset_index().sort_values("folder_count", ascending=False)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Folders per 1-Level SNC","Documents per 1-Level SNC"), shared_yaxes=True)
    fig.add_trace(go.Bar(y=primary_stats["snc_primary"], x=primary_stats["folder_count"], orientation="h",
        marker_color=_DA_COLORS["primary"], name="Folders"), row=1, col=1)
    fig.add_trace(go.Bar(y=primary_stats["snc_primary"], x=primary_stats["total_docs"], orientation="h",
        marker_color="#2563EB", name="Documents"), row=1, col=2)
    fig = _apply_light_theme(fig, max(400, len(primary_stats) * 22))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    # ── SECTION 2: SNC Distribution ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 🏷️ SNC Distribution")

    # Build count maps used across tabs
    snc3_count_map = folders_df.groupby("snc_3level").size().to_dict()
    snc2_count_map = folders_df.groupby("snc_2level").size().to_dict()
    snc1_count_map = folders_df.groupby("snc_primary").size().to_dict()

    tab_3l, tab_2l, tab_1l, tab_deepdive = st.tabs([
        "3-Level SNC", "2-Level SNC", "1-Level SNC", "🔎 SNC Deep Dive"
    ])

    with tab_3l:
        snc3 = folders_df.groupby("snc_3level").agg(
            folder_count=("folder_id","count"), total_docs=("doc_count","sum")
        ).reset_index().sort_values("folder_count", ascending=False)
        st.metric("Distinct 3-Level SNCs", len(snc3))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top-10 by Folder Count")
            top10 = snc3.head(10).copy()
            st.dataframe(top10[["snc_3level","folder_count","total_docs"]].rename(
                columns={"snc_3level":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True)
        with c2:
            st.markdown("#### Bottom-10 by Folder Count")
            bot10 = snc3.tail(10).copy()
            st.dataframe(bot10[["snc_3level","folder_count","total_docs"]].rename(
                columns={"snc_3level":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True)
        fig = px.histogram(snc3, x="folder_count", nbins=40, color_discrete_sequence=[_DA_COLORS["primary"]],
            labels={"folder_count":"Folders","count":"SNCs"}, title="Histogram: Folders per SNC (3-Level)")
        st.plotly_chart(_apply_light_theme(fig), width="stretch")
        st.markdown("#### Top 40 SNCs (3-Level) by Folder Count")
        fig_top40 = px.bar(snc3.head(40), x="snc_3level", y="folder_count",
            color_discrete_sequence=[_DA_COLORS["primary"]],
            labels={"snc_3level":"SNC (3-Level)","folder_count":"Folders"},
            title="Top 40 SNCs (3-Level)")
        fig_top40.update_xaxes(tickangle=45)
        st.plotly_chart(_apply_light_theme(fig_top40, 450), width="stretch")
        with st.expander("All SNCs — Complete Table"):
            st.dataframe(snc3.rename(columns={"snc_3level":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True, height=400)

    with tab_2l:
        snc2 = folders_df.groupby("snc_2level").agg(
            folder_count=("folder_id","count"), total_docs=("doc_count","sum")
        ).reset_index().sort_values("folder_count", ascending=False)
        st.metric("Distinct 2-Level SNCs", len(snc2))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top-10 by Folder Count")
            st.dataframe(snc2.head(10).rename(columns={"snc_2level":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True)
        with c2:
            st.markdown("#### Bottom-10 by Folder Count")
            st.dataframe(snc2.tail(10).rename(columns={"snc_2level":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True)
        fig = px.histogram(snc2, x="folder_count", nbins=30, color_discrete_sequence=[_DA_COLORS["primary"]],
            labels={"folder_count":"Folders","count":"SNCs"}, title="Histogram: Folders per SNC (2-Level)")
        st.plotly_chart(_apply_light_theme(fig), width="stretch")
        fig_top40 = px.bar(snc2.head(40), x="snc_2level", y="folder_count",
            color_discrete_sequence=[_DA_COLORS["primary"]],
            labels={"snc_2level":"SNC (2-Level)","folder_count":"Folders"},
            title="Top 40 SNCs (2-Level)")
        fig_top40.update_xaxes(tickangle=45)
        st.plotly_chart(_apply_light_theme(fig_top40, 450), width="stretch")
        with st.expander("All SNCs — Complete Table"):
            st.dataframe(snc2.rename(columns={"snc_2level":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True, height=400)

    with tab_1l:
        snc1 = folders_df.groupby("snc_primary").agg(
            folder_count=("folder_id","count"), total_docs=("doc_count","sum")
        ).reset_index().sort_values("folder_count", ascending=False)
        st.metric("Distinct 1-Level SNC Codes", len(snc1))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top-10 by Folder Count")
            st.dataframe(snc1.head(10).rename(columns={"snc_primary":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True)
        with c2:
            st.markdown("#### Bottom-10 by Folder Count")
            st.dataframe(snc1.tail(10).rename(columns={"snc_primary":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True)
        fig = px.histogram(snc1, x="folder_count", nbins=20, color_discrete_sequence=[_DA_COLORS["primary"]],
            labels={"folder_count":"Folders","count":"SNCs"}, title="Histogram: Folders per 1-Level SNC")
        st.plotly_chart(_apply_light_theme(fig), width="stretch")
        fig_top40 = px.bar(snc1, x="snc_primary", y="folder_count",
            color_discrete_sequence=[_DA_COLORS["primary"]],
            labels={"snc_primary":"1-Level SNC","folder_count":"Folders"},
            title="Folders per 1-Level SNC Code (All)")
        fig_top40.update_xaxes(tickangle=45)
        st.plotly_chart(_apply_light_theme(fig_top40, 450), width="stretch")
        with st.expander("All SNCs — Complete Table"):
            st.dataframe(snc1.rename(columns={"snc_primary":"SNC","folder_count":"Folders","total_docs":"Documents"}),
                width="stretch", hide_index=True, height=400)

    with tab_deepdive:
        snc_options = sorted(folders_df["snc_3level"].unique().tolist())
        snc_labels_map = folders_df.groupby("snc_3level")["label"].first().to_dict()
        # Dropdown: SNC — Label — Count of folders: N
        snc_display = [
            f"{s} — {snc_labels_map.get(s,'')} — Count of folders: {snc3_count_map.get(s,0)}"
            for s in snc_options
        ]
        sel_idx = st.selectbox("Select SNC Code (SNC - Folder Label - Number of folders with this SNC)", range(len(snc_options)), format_func=lambda i: snc_display[i],
            key="deepdive_snc_select")
        selected_snc = snc_options[sel_idx]

        snc_folders = folders_df[folders_df["snc_3level"] == selected_snc].copy()
        snc_folder_ids = set(snc_folders["folder_id"].tolist())
        snc_docs = items_df[items_df["folder_id"].isin(snc_folder_ids)]

        expanded = snc_folders["label_parent_expanded"].iloc[0] if len(snc_folders) > 0 else "N/A"
        st.markdown(f"## {selected_snc} — {expanded}")

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("📁 Folders", len(snc_folders)); cc2.metric("📄 Documents", len(snc_docs))
        cc3.metric("📦 Boxes", snc_folders["box"].nunique())
        cc4.metric("Avg Docs/Folder", f"{snc_folders['doc_count'].mean():.1f}" if len(snc_folders) > 0 else "0")

        dd_tab1, dd_tab2 = st.tabs(["📁 Folders & Docs", "📊 Keywords & Themes"])
        with dd_tab1:
            folder_display = snc_folders[["folder_id","box","label","date","endDate","doc_count"]].copy()
            folder_display = folder_display.sort_values("doc_count", ascending=False)
            st.dataframe(folder_display.rename(columns={"folder_id":"Folder ID","box":"Box","label":"Label","date":"Start","endDate":"End","doc_count":"Docs"}),
                width="stretch", hide_index=True, height=350)
            st.markdown("### Browse Folder Documents")
            folder_ids = snc_folders["folder_id"].tolist()
            folder_labels_map = snc_folders.set_index("folder_id")["label"].to_dict()
            sel_folder = st.selectbox("Select Folder (Folder ID - SNC - Folder Label)", folder_ids, format_func=lambda x: f"{x} — {folder_labels_map.get(x,'')}", key="deepdive_folder_select")
            if sel_folder:
                folder_docs_df = snc_docs[snc_docs["folder_id"] == sel_folder]
                st.markdown(f"**{len(folder_docs_df)} documents in this folder:**")
                for _, doc in folder_docs_df.iterrows():
                    with st.expander(f"📄 {doc['doc_id']} — {doc['title']}"):
                        if doc["has_summary"]: st.markdown(f"**Summary:** {doc['summary']}")
                        st.caption(f"Date: {doc['date']} | OCR Pages: {doc['ocr_pages']}")
        with dd_tab2:
            st.markdown("### Top Keywords from Documents Titles")
            title_kw = _extract_keywords(snc_docs["title"].dropna().tolist(), top_n=25)
            if title_kw:
                kw_df = pd.DataFrame(title_kw, columns=["Word","Count"])
                fig = px.bar(kw_df, x="Count", y="Word", orientation="h",
                    color_discrete_sequence=[_DA_COLORS["primary"]])
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(_apply_light_theme(fig, 500), width="stretch")
            st.markdown("### Top Keywords from Summaries")
            sum_kw = _extract_keywords(snc_docs["summary"].dropna().tolist(), top_n=25)
            if sum_kw:
                kw_df2 = pd.DataFrame(sum_kw, columns=["Word","Count"])
                fig2 = px.bar(kw_df2, x="Count", y="Word", orientation="h",
                    color_discrete_sequence=[_DA_COLORS["primary"]])
                fig2.update_yaxes(autorange="reversed")
                st.plotly_chart(_apply_light_theme(fig2, 500), width="stretch")

    # ── SECTION 3: Document Analysis ─────────────────────────────────
    st.markdown("---")
    st.markdown("## 📄 Document Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", f"{len(items_df):,}"); col2.metric("With Titles", f"{items_df['has_title'].sum():,}")
    col3.metric("With Summaries", f"{items_df['has_summary'].sum():,}"); col4.metric("Avg OCR Pages", f"{items_df['ocr_pages'].mean():.1f}")

    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["📊 Statistics", "🔍 Browse by SNC", "🗂️ Browse by Folder"])

    with doc_tab1:
        ca, cb = st.columns(2)
        with ca:
            fig = px.histogram(items_df, x="ocr_pages", nbins=50, color_discrete_sequence=[_DA_COLORS["primary"]],
                labels={"ocr_pages":"OCR Pages","count":"Documents"}, title="OCR Pages per Document")
            st.plotly_chart(_apply_light_theme(fig), width="stretch")
        with cb:
            dpf = items_df.groupby("folder_id").size().reset_index(name="doc_count")
            fig = px.histogram(dpf, x="doc_count", nbins=50, color_discrete_sequence=[_DA_COLORS["primary"]],
                labels={"doc_count":"Documents per Folder","count":"Folders"}, title="Documents per Folder")
            st.plotly_chart(_apply_light_theme(fig), width="stretch")
        st.markdown("### Most Common Words in Document Titles")
        title_kws = _extract_keywords(items_df["title"].dropna().tolist(), top_n=30)
        if title_kws:
            kw_df = pd.DataFrame(title_kws, columns=["Word","Count"])
            fig = px.bar(kw_df, x="Count", y="Word", orientation="h",
                color_discrete_sequence=[_DA_COLORS["primary"]],
                title="Top 30 Words in Document Titles")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(_apply_light_theme(fig, 600), width="stretch")

    with doc_tab2:
        # Build dropdown: SNC — Label — Count of folders: N
        snc_opts = sorted(folders_df["snc_3level"].unique().tolist())
        snc_labels_map_da = folders_df.groupby("snc_3level")["label"].first().to_dict()
        snc_display_da = [
            f"{s} — {snc_labels_map_da.get(s,'')} — Count of folders: {snc3_count_map.get(s,0)}"
            for s in snc_opts
        ]
        sel_snc_idx = st.selectbox("Select SNC Code (SNC - Folder Label - Number of folders with this SNC)", range(len(snc_opts)), format_func=lambda i: snc_display_da[i],
            index=0, key="doc_browse_snc")
        sel_snc = snc_opts[sel_snc_idx]
        snc_flds = folders_df[folders_df["snc_3level"] == sel_snc]
        snc_fld_ids = set(snc_flds["folder_id"].tolist())
        snc_docs2 = items_df[items_df["folder_id"].isin(snc_fld_ids)]
        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Folders in SNC", len(snc_flds))
        cb2.metric("Documents in SNC", len(snc_docs2))
        with cb3:
            expanded_txt = snc_flds["label_parent_expanded"].iloc[0] if len(snc_flds) > 0 else "N/A"
            st.markdown(
                f'<div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 16px; min-height: 82px;">'
                f'<div style="color: #64748B; font-size: 0.85rem;">Expanded Label</div>'
                f'<div style="color: #0F172A; font-size: 0.88rem; font-weight: 600; margin-top: 4px; line-height: 1.3;">{expanded_txt}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        snc_texts = snc_docs2["title"].dropna().tolist() + snc_docs2["summary"].dropna().tolist()
        snc_kws = _extract_keywords(snc_texts, top_n=20)
        if snc_kws:
            kw_df3 = pd.DataFrame(snc_kws, columns=["Word","Count"])
            fig = px.bar(kw_df3, x="Count", y="Word", orientation="h",
                color_discrete_sequence=[_DA_COLORS["secondary"]])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(_apply_light_theme(fig, 450), width="stretch")
        st.markdown("### Folders")
        for _, folder_row in snc_flds.iterrows():
            fid = folder_row["folder_id"]; n_docs2 = folder_row["doc_count"]
            with st.expander(f"📁 {folder_row['label']} ({n_docs2} docs) — ID: {fid}"):
                fitems = snc_docs2[snc_docs2["folder_id"] == fid]
                for _, doc_row in fitems.head(20).iterrows():
                    st.markdown(f"**📄 {doc_row['doc_id']}** — {doc_row['title']}")
                    if doc_row["has_summary"]:
                        s = str(doc_row.get("summary",""))
                        st.caption(s[:300] + "…" if len(s) > 300 else s)
                    st.markdown("---")
                if len(fitems) > 20: st.info(f"Showing first 20 of {len(fitems)} documents.")

    with doc_tab3:
        # Browse By Folder — all folders, dropdown: FOLDER_ID — SNC — Label
        all_folder_ids = sorted(folders_df["folder_id"].tolist())
        folder_snc_map = folders_df.set_index("folder_id")["snc_clean"].to_dict()
        folder_label_map = folders_df.set_index("folder_id")["label"].to_dict()
        folder_doc_count_map = folders_df.set_index("folder_id")["doc_count"].to_dict()
        folder_box_map = folders_df.set_index("folder_id")["box"].to_dict()
        folder_date_map = folders_df.set_index("folder_id")["date"].to_dict()
        folder_expanded_map = folders_df.set_index("folder_id")["label_parent_expanded"].to_dict()

        folder_display_labels = [
            f"{fid} — {folder_snc_map.get(fid,'')} — {folder_label_map.get(fid,'')}"
            for fid in all_folder_ids
        ]
        sel_folder_idx = st.selectbox(
            "Select Folder (Folder ID - SNC - Folder Label)",
            range(len(all_folder_ids)),
            format_func=lambda i: folder_display_labels[i],
            key="da_browse_folder_select"
        )
        sel_folder_id = all_folder_ids[sel_folder_idx]
        sel_folder_docs = items_df[items_df["folder_id"] == sel_folder_id]

        # Folder metadata summary
        fb1, fb2, fb3, fb4 = st.columns(4)
        fb1.metric("📦 Box", folder_box_map.get(sel_folder_id, "N/A"))
        fb2.metric("🏷️ SNC", folder_snc_map.get(sel_folder_id, "N/A"))
        fb3.metric("📄 Documents", folder_doc_count_map.get(sel_folder_id, 0))
        fb4.metric("📅 Date", folder_date_map.get(sel_folder_id, "N/A"))

        parent_exp = folder_expanded_map.get(sel_folder_id, "")
        if parent_exp:
            st.markdown(f"**Classification:** {parent_exp}")

        st.markdown(f"### Documents in Folder `{sel_folder_id}`")
        if sel_folder_docs.empty:
            st.caption("No documents indexed for this folder.")
        else:
            for _, doc in sel_folder_docs.iterrows():
                with st.expander(f"📄 {doc['doc_id']} — {doc['title']}"):
                    if doc["has_summary"]: st.markdown(f"**Summary:** {doc['summary']}")
                    st.caption(f"Date: {doc['date']} | OCR Pages: {doc['ocr_pages']}")







# ============================================================
# TOPIC VIEWER UI (Enhanced — Hierarchical)
# ============================================================

def run_topic_viewer_ui():
    folders_meta, items_meta = u2.load_metadata()
    ecf_data = u2.load_ecf_data()
    q_docs = u2.load_qrels_data(u2.PATH_QRELS_DOCS)
    q_folders = u2.load_qrels_data(u2.PATH_QRELS_FOLDERS)
    q_boxes = u2.load_qrels_data(u2.PATH_QRELS_BOXES)

    st.title("🔍 Task Viewer")
    st.caption("⭐ **Relevance Legend:** ⭐⭐⭐ = Highly Relevant (Grade 3) | ⭐ = Relevant (Grade 1)")

    all_topics = {}
    if "ExperimentSets" in ecf_data:
        for es in ecf_data["ExperimentSets"]:
            if "Topics" in es: all_topics.update(es["Topics"])

    topic_ids = sorted(all_topics.keys())
    # Topic selector in main page (not sidebar), showing number + title
    topic_display = []
    for tid in topic_ids:
        match = re.search(r'\d+$', tid)
        num = int(match.group()) if match else tid
        title = all_topics[tid].get('TITLE', '')
        topic_display.append(f"{num} — {title}")

    sel_idx = st.selectbox("Select Topic:", range(len(topic_ids)), format_func=lambda i: topic_display[i])
    sel_topic = topic_ids[sel_idx]

    if not sel_topic: return

    t_data = all_topics[sel_topic]
    match = re.search(r'\d+$', sel_topic)
    topic_num = int(match.group()) if match else sel_topic

    with st.expander("Topic Details", expanded=True):
        st.subheader(f"Topic {topic_num}")
        st.caption(f"ID: `{sel_topic}`")
        st.info(f"**Title:** {t_data.get('TITLE','')}")
        c1, c2 = st.columns([1, 1])
        c1.info(f"**Description:**\n{t_data.get('DESCRIPTION','')}")
        c2.warning(f"**Narrative:**\n{t_data.get('NARRATIVE','')}")

    st.divider()

    # Collect all relevant items by grade
    rd = q_docs.get(sel_topic, [])
    rf = q_folders.get(sel_topic, [])
    rb = q_boxes.get(sel_topic, [])

    # Build hierarchical structure: Box → Folder → Document
    # Maps
    doc_grades = {did: sc for did, sc in rd}
    folder_grades = {fid: sc for fid, sc in rf}
    box_grades = {bid: sc for bid, sc in rb}

    # Group folders by box
    box_to_folders = defaultdict(set)
    for fid, _ in rf:
        f_meta = folders_meta.get(fid, {})
        box_id = f_meta.get("box", "Unknown")
        box_to_folders[box_id].add(fid)

    # Group documents by folder
    folder_to_docs = defaultdict(set)
    for did, _ in rd:
        d_meta = items_meta.get(did, {})
        fid = d_meta.get("Sushi Folder", "Unknown")
        folder_to_docs[fid].add(did)
        # Also ensure the folder's box is captured
        f_meta = folders_meta.get(fid, {})
        box_id = f_meta.get("box", "Unknown")
        box_to_folders[box_id].add(fid)

    # Also add boxes from qrels that might not have folders
    for bid, _ in rb:
        if bid not in box_to_folders:
            box_to_folders[bid] = set()

    # Summary metrics
    st.markdown(f"**Relevant Items:** {len(rb)} boxes, {len(rf)} folders, {len(rd)} documents")

    # Render hierarchical tree
    for box_id in sorted(box_to_folders.keys()):
        b_grade = box_grades.get(box_id, 0)
        b_stars = u3.grade_stars(b_grade) if b_grade > 0 else ""
        folder_ids_in_box = box_to_folders[box_id]

        with st.expander(f"📦 Box {box_id} {b_stars}", expanded=True):
            if not folder_ids_in_box:
                st.caption("No relevant folders in this box.")
                continue

            for fid in sorted(folder_ids_in_box):
                f_grade = folder_grades.get(fid, 0)
                f_stars = u3.grade_stars(f_grade)
                f_meta = folders_meta.get(fid, {})
                label = f_meta.get("label", "N/A")
                snc = f_meta.get("snc", "N/A")
                parent_exp = f_meta.get("label_parent_expanded", "")

                with st.expander(f"📂 {fid} — {label} {f_stars}", expanded=False):
                    st.markdown(f"**SNC:** `{snc}` | **Label:** {label}")
                    if parent_exp:
                        st.markdown(f"**Parent Classification:** {parent_exp}")
                    scope = f_meta.get("raw_scope", "")
                    if scope and str(scope) != 'nan':
                        st.markdown(f"**Scope Note:** {str(scope)[:500]}")
                    st.caption(f"Box: {f_meta.get('box','N/A')} | Date: {f_meta.get('date','N/A')}")

                    # Documents in this folder
                    docs_in_folder = folder_to_docs.get(fid, set())
                    if docs_in_folder:
                        st.markdown(f"**{len(docs_in_folder)} relevant document(s):**")
                        for did in sorted(docs_in_folder):
                            d_grade = doc_grades.get(did, 0)
                            d_stars = u3.grade_stars(d_grade)
                            d_meta = items_meta.get(did, {})
                            title = u2.get_smart_title(d_meta, did)

                            with st.expander(f"📄 {did} — {title} {d_stars}", expanded=False):
                                summary = d_meta.get("summary", "")
                                if summary and str(summary) != 'nan':
                                    st.info(f"**Summary:** {summary}")
                                else:
                                    st.caption("No summary available.")

                                ocr_pages = d_meta.get("ocr", [])
                                if isinstance(ocr_pages, list) and len(ocr_pages) > 0:
                                    first_page = ocr_pages[0]
                                    if first_page and str(first_page) != 'nan':
                                        with st.expander("📃 First Page OCR Text", expanded=False):
                                            st.text(str(first_page)[:2000])
                                else:
                                    st.caption("No OCR text available.")
                    else:
                        st.caption("No relevant documents in this folder.")


# ============================================================
# HOW-TO GUIDE
# ============================================================

def run_howto_ui():
    st.title("📖 How-To Guide — SUSHI BAR")

    st.markdown("""
## What is the SUSHI Task?

The **SUSHI (Searching Unseen Sources for Historical Information)** task focuses on
**folder-level ranking** in sparsely digitized archives. The collection consists of U.S. State Department records
on Brazil from the 1960s–1970s, organized in a physical hierarchy: **Boxes → Folders → Documents**.

The challenge: only **~5 documents per box** have been digitized (with metadata like title, summary, OCR text).
Most folders have **no digitized documents** at all. The task is to rank folders by relevance to a research topic,
despite this severe data sparsity.

---

## Key Concepts

### 📦 Box / 📂 Folder / 📄 Document Hierarchy

| Level | Count | Description |
|-------|-------|-------------|
| **Box** | ~126 | Physical boxes storing folders. Each box contains 5–20+ folders. |
| **Folder** | 1,336 | The unit of retrieval. Has a label, SNC code, dates, and optionally a scope note. |
| **Document** | 31,681 | Individual records (cables, memos, reports). Only ~630 per Training Set are "digitized." |

### 🏷️ SNC (Subject-Numeric Code)

The U.S. State Department's subject classification system used to organize diplomatic records. Each folder in the collection can have an SNC code attached to it.

In the **Collection Viewer**, SNC codes can be explored across **three levels of granularity**:

- **1-Level SNC**:
  The top-level subject area represented by a 3-letter alphabetic prefix (e.g., `POL` for Political Affairs, `AGR` for Agriculture, `DEF` for Defense Affairs, `LAB` for Labor & Manpower, `SCI` for Science & Technology).
  - *Broadest aggregation*: Groups all folders under primary domains (`POL` alone covers 561 folders across many sub-topics).

- **2-Level SNC**:
  The primary category combined with the major numeric topic code (e.g., `POL 15` for Government/Elections, `AGR 12` for Agricultural Production).
  - *Intermediate grouping*: Clusters closely related thematic sub-topics together.

- **3-Level SNC**:
  The complete, fine-grained classification code including numeric extensions and qualifiers (e.g., `POL 15-1` for Specific Election Reports).
  - *Finest granularity*: Provides exact subject specificity across distinct individual codes in the collection.

### 🧪 Training Sets

Defines which documents are "digitized" (available as training data) for an experiment.
- **Uniform Training Sets**: 5 documents randomly sampled per box (~630 total).
- **Skewed Training Sets**: Uneven sampling across boxes.
- **All Docs**: All 31,681 documents available (upper bound only).

Different seeds produce different random samples, so experiments are run across multiple seeds and averaged.

### 📊 nDCG@5 (Normalized Discounted Cumulative Gain at 5)

The primary evaluation metric used across all experiments. It measures how effectively an IR model ranks folders that contain one or more relevant documents within the top-5 positions.

- **How nDCG@5 is calculated:**
  - **Relevance Gain:** Higher relevance grades (e.g. Grade 3 for Highly Relevant vs Grade 1 for Relevant) yield higher gain values for folders containing relevant documents.
  - **Position Discount:** Gain is logarithmically discounted based on rank position, placing greater value on placing folders that contain one or more relevant documents at ranks 1–2 than ranks 4–5.
  - **Normalization:** The resulting Discounted Cumulative Gain (DCG@5) is divided by the Ideal DCG (IDCG@5)—the maximum possible score achievable if folders that contain one or more relevant documents were placed in optimal order at the top.
- **Score Interpretation:**
  - **1.0** = Perfect ranking (the top-5 positions contain folders with relevant documents in optimal order).
  - **0.0** = No folders with relevant documents appear within the top-5 retrieved positions.

### ⭐ Relevance Grades

| Grade | Meaning | Description | Display |
|-------|---------|-------------|---------| 
| **3** | Highly Relevant | Folder contains highly relevant documents for the topic | ⭐⭐⭐ |
| **1** | Relevant | Folder contains relevant documents for the topic | ⭐ |
| **0** | Not Relevant | Folder contains no relevant documents for the topic | — |

---

## App Pages


### 📦 Collection Viewer
Explore the SUSHI collection structure:
- **SNC Distribution tabs** (3-Level / 2-Level / 1-Level): For each SNC granularity, see the count of distinct codes, top-10 / bottom-10 by folder count, a histogram, a top-40 bar chart, and a complete table.
- **SNC Deep Dive**: Pick an SNC code from the dropdown (shows `SNC — Label — Count of folders: N`) to explore all its folders, browse documents, and view keyword clouds.
- **Document Analysis → Browse by SNC**: Filter documents by SNC code to see keyword summaries and folder listings.
- **Document Analysis → Browse by Folder**: Pick any of the 1,336 folders by `FOLDER_ID — SNC — Label` to see its metadata and all its documents.

### 🔍 Task Viewer
Browse the 45 evaluation topics (T1–T45). Select a topic to see its description and narrative, then explore the hierarchical tree of relevant **Boxes → Folders → Documents** with star-based relevance grades.

### 🧪 Training Set Viewer
Analyze what the Training Set includes:
- **Overview**: Histogram of documents per covered folder.
- **By SNC**: Coverage table by SNC code with KPIs for SNCs with/without docs.
- **By Box**: Coverage table sorted by coverage %.
- **By Folder**: Browse only the folders covered by this Training Set (dropdown: `FOLDER_ID — SNC — Label`).
- **Relevance Coverage**: Cross-reference Training Set coverage with the 45-topic qrels, including % coverage for Grade 3 and Grade 1 relevant folders.
- **Compare Training Sets**: Overlay two Training Sets to compare their coverage distributions.

### 🔬 Single Experiment Viewer
Select an experiment configuration to compare models (e.g., BM25 vs ColBERT) within it:
- **Global Performance**: KPI metrics showing mean nDCG@5 ± 95% margin per model.
- **Model Comparison Chart**: Interactive dumbbell chart showing per-topic mean nDCG@5 and confidence intervals.
- **Seed Variance**: Horizontal box plot displaying nDCG@5 distribution across seeds per topic.

### ⚔️ Two-Experiment Viewer
Direct side-by-side comparison of any two experiment runs across any sets of experiments:
- **Side-by-Side KPIs & Delta**: Global mean nDCG@5 metrics and exact performance delta (A - B).
- **Wilcoxon Signed-Rank Test**: Statistical significance test results (p-value, winner, win counts).
- **Overlay Comparison Chart**: Side-by-side topic overlay dumbbell chart comparing Run A (Blue) vs Run B (Orange).
- **Topic Separation Analysis**: Categorized breakdown of topics into Better (≥ +10%), About Equal (within ±10%), and Worse (≤ -10%).

---

## Data Dictionary

### Folder Metadata (`FoldersV1.3.json`)

| Field | Type | Fill Rate | Description |
|-------|------|-----------|-------------|
| `box` | string | 100% | Physical box identifier |
| `snc` | string | 95% | Subject-Numeric Code (68 folders = "Unknown") |
| `label` | string | 100% | Original folder label |
| `date` / `endDate` | string | 100% / 25% | Date range (75% have Unknown end date) |
| `main_title` | string | 100% | Human-readable SNC meaning |
| `label_parent_expanded` | string | 86% | Full hierarchical semantic path |
| `raw_scope` | string | 38% | Scope note text |

### Document Metadata (`itemsV1.2.json`)

| Field | Type | Description |
|-------|------|-------------|
| `Sushi Box` / `Sushi Folder` | string | Parent box and folder IDs |
| `title` | string | Document title |
| `date` | string | Document date |
| `summary` | string | GPT-4o generated summary |
| `ocr` | list[string] | OCR text per page |

---

### 🧪 Experiment Naming Conventions (5-Element Dotted Notation)

Run folders in `all_runs/` follow the structured 5-element dotted pattern:
`{Sample}.{Ranker}.{Fields}.{LabelSearch}.{ScorePropagation}`

#### 1. Sample (`Sample`)
- **`U`**: Uniform random sampling (~630 digitized docs, 5 per box)
- **`K`**: Skewed non-uniform sampling (~630 digitized docs)
- **`A`**: All 31,681 documents indexed (upper bound ceiling)

#### 2. Ranker / Model (`Ranker`)
- **`B`**: BM25F
- **`C`**: ColBERT late-interaction
- **`E`**: Dense Embedding Similarity (`all-mpnet-base-v2`)
- **`W`**: Reciprocal Rank Fusion (RRF) of Weighted BM25 + ColBERT + Embeddings
- **`X`**: RRF of BM25 + ColBERT
- **`Y`**: RRF of BM25 + Embeddings
- **`Z`**: RRF of BM25 + ColBERT + Embeddings

#### 3. Fields (`Fields`, 4-character string: `T`, `O`, `F`, `S`)
- **`T`**: Title, **`O`**: OCR, **`F`**: Folder Label, **`S`**: Summary
- Unused fields are represented with dashes (`-`).
- Examples: `T---` (Title only), `-O--` (OCR only), `--F-` (Folder Label only), `---S` (Summary only), `T-FS` (Title+Folder+Summary), `TOF-` (Title+OCR+Folder), `TOFS` (All document fields), `----` (No doc fields, label search only).

#### 4. Label Search (`LabelSearch`)
- **`L`**: Weighted RRF with full-collection folder Label search
- **`x`**: No label search

#### 5. Score Propagation (`ScorePropagation`)
- **`1`**: Same SNC code score propagation depth 1
- **`2`**: Same SNC code score propagation depth 2
- **`x`**: No score propagation

#### Folder Naming Examples
- **`U.B.T---.x.x`**: Uniform sample, BM25 ranker, Title field only, no label search, no score propagation.
- **`U.W.TOFS.L.2`**: Uniform sample, Weighted RRF ranker, all document fields, Label search enabled, propagation depth 2.
- **`K.Z.TOFS.x.x`**: Skewed sample, RRF(B+C+E) ranker, all document fields, no label search, no score propagation.
- **`A.Y.T-FS.x.x`**: All documents sample, RRF(B+E) ranker, Title+Folder+Summary fields, no label search, no score propagation.
""")


# ============================================================
# ECF INSPECTOR
# ============================================================

def run_ecf_inspector_ui():
    st.title("🧪 Training Set Viewer")
    st.caption("Analyze Training Set coverage: which folders, SNCs, and boxes have digitized documents.")

    folders_meta = u4._load_folders_meta()
    all_ecfs = u4.list_available_ecfs()

    if not all_ecfs:
        st.error("No Training Set files found.")
        return

    # Type filter + Training Set selection
    ecf_types = sorted(set(e["type"] for e in all_ecfs))
    col_type, col_ecf_a, col_ecf_b = st.columns([1, 2, 2])

    with col_type:
        sel_type = st.selectbox("Training Set Sampling Type:", ["All"] + ecf_types)

    filtered_ecfs = [e for e in all_ecfs if sel_type == "All" or e["type"] == sel_type]

    with col_ecf_a:
        ecf_a_labels = [e["label"] for e in filtered_ecfs]
        sel_ecf_a_idx = st.selectbox("Select Training Set:", range(len(filtered_ecfs)),
            format_func=lambda i: ecf_a_labels[i], key="ecf_a")
        ecf_a = filtered_ecfs[sel_ecf_a_idx]

    with col_ecf_b:
        compare_ecfs = [e for e in filtered_ecfs if e["path"] != ecf_a["path"]]
        ecf_b_options = ["(None — no comparison)"] + [e["label"] for e in compare_ecfs]
        sel_ecf_b_idx = st.selectbox("Compare against:", range(len(ecf_b_options)),
            format_func=lambda i: ecf_b_options[i], key="ecf_b")
        ecf_b = compare_ecfs[sel_ecf_b_idx - 1] if sel_ecf_b_idx > 0 else None

    # Load ECF A
    docs_a = u4.load_ecf_training_docs(ecf_a["path"])
    parsed_a = u4.parse_ecf_docs(docs_a)
    metrics_a = u4.compute_headline_metrics(parsed_a, folders_meta)

    # Headline metrics — clean non-truncating format with percentage text right below count (no ? tooltips)
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 16px; margin-bottom:20px;">'
            f'<div style="color:#64748B; font-size:0.85rem; font-weight:600;">📄 Documents Included</div>'
            f'<div style="color:#0F172A; font-weight:700; font-size:1.25rem; margin-top:2px;">{metrics_a["docs_included"]:,} / {metrics_a["docs_total"]:,}</div>'
            f'<div style="color:#64748B; font-size:0.85rem; margin-top:2px;">{metrics_a["docs_pct"]}% of total</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with mc2:
        st.markdown(
            f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 16px; margin-bottom:20px;">'
            f'<div style="color:#64748B; font-size:0.85rem; font-weight:600;">📂 Folders with ≥1 Doc</div>'
            f'<div style="color:#0F172A; font-weight:700; font-size:1.25rem; margin-top:2px;">{metrics_a["folders_covered"]:,} / {metrics_a["folders_total"]:,}</div>'
            f'<div style="color:#64748B; font-size:0.85rem; margin-top:2px;">{metrics_a["folders_pct"]}% covered</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with mc3:
        st.markdown(
            f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 16px; margin-bottom:20px;">'
            f'<div style="color:#64748B; font-size:0.85rem; font-weight:600;">📦 Boxes Represented</div>'
            f'<div style="color:#0F172A; font-weight:700; font-size:1.25rem; margin-top:2px;">{metrics_a["boxes_covered"]:,} / {metrics_a["boxes_total"]:,}</div>'
            f'<div style="color:#64748B; font-size:0.85rem; margin-top:2px;">{metrics_a["boxes_pct"]}% covered</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with mc4:
        st.markdown(
            f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 16px; margin-bottom:20px;">'
            f'<div style="color:#64748B; font-size:0.85rem; font-weight:600;">📊 Avg Docs/Covered Folder</div>'
            f'<div style="color:#0F172A; font-weight:700; font-size:1.25rem; margin-top:2px;">{metrics_a["avg_docs_per_folder"]:.2f}</div>'
            f'<div style="color:#64748B; font-size:0.85rem; margin-top:2px;">&nbsp;</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Tabs — added "📂 By Folder" between Box and Relevance Coverage
    tab_names = ["📊 Overview", "🏷️ By SNC", "📂 By Folder", "📦 By Box", "🎯 Relevance Coverage"]
    if ecf_b:
        tab_names.append("⚖️ Compare Training Sets")

    tabs = st.tabs(tab_names)

    # Tab 1: Overview
    with tabs[0]:
        st.subheader("Documents per Covered Folder")
        fdc = parsed_a.get("folder_doc_counts", {})
        if fdc:
            fdc_df = pd.DataFrame(list(fdc.items()), columns=["Folder", "Doc Count"])
            fig = px.histogram(fdc_df, x="Doc Count", nbins=30, color_discrete_sequence=[_DA_COLORS["primary"]],
                labels={"Doc Count": "Documents per Folder", "count": "Folders"})
            st.plotly_chart(_apply_light_theme(fig, 400), width="stretch")

    # Tab 2: By SNC — removed stacked bar, added "SNCs with No Docs" KPI
    with tabs[1]:
        snc_level = st.radio("SNC Granularity:", ["1-Level SNC", "3-Level SNC"], horizontal=True)
        if snc_level == "1-Level SNC":
            snc_df = u4.compute_snc_coverage(parsed_a, folders_meta)
        else:
            snc_df = u4.compute_snc_3level_coverage(parsed_a, folders_meta)

        snc_without_docs = int((snc_df["Covered Folders"] == 0).sum())
        snc_with_docs = int((snc_df["Covered Folders"] > 0).sum())
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("✅ SNCs with ≥1 Doc", snc_with_docs)
        kpi2.metric("🚫 SNCs with No Docs", snc_without_docs)

        st.subheader("SNC Coverage — Table")
        st.dataframe(snc_df.sort_values("Coverage %", ascending=True), width="stretch", hide_index=True, height=400)

    # Tab 4: By Folder — only training set-covered folders, dropdown: FOLDER_ID — SNC — Label
    with tabs[2]:
        folder_detail_df = u4.compute_folder_detail(parsed_a, folders_meta)
        if folder_detail_df.empty:
            st.info("No folders covered by this Training Set.")
        else:
            st.caption(f"**{len(folder_detail_df)}** folders covered by this Training Set (have ≥1 training document).")
            folder_dropdown_labels = folder_detail_df["Dropdown Label"].tolist()
            sel_folder_ecf_idx = st.selectbox(
                "Select Folder (Folder ID - SNC - Folder Label)",
                range(len(folder_detail_df)),
                format_func=lambda i: folder_dropdown_labels[i],
                key="ecf_folder_select"
            )
            row = folder_detail_df.iloc[sel_folder_ecf_idx]

            if row["Label Parent Expanded"]:
                st.markdown(f"**Classification:** {row['Label Parent Expanded']}")
            if row["Scope Note"]:
                st.info(f"**Scope Note:** {row['Scope Note']}")

            ef1, ef2, ef3 = st.columns(3)
            ef1.metric("📦 Box", row["Box"])
            ef2.metric("🏷️ SNC", row["SNC"])
            docs_col_name = "Docs in Training Set" if "Docs in Training Set" in row else "Docs in ECF"
            ef3.metric("📄 Docs in Training Set", int(row[docs_col_name]))

            # List documents in folder with title and summary
            sel_folder_id = row["Folder ID"]
            folder_doc_ids = parsed_a.get("folder_doc_ids", {}).get(sel_folder_id, [])
            items_meta = u4._load_items_meta()

            st.markdown("---")
            st.subheader(f"📄 Documents in Training Set for Folder `{sel_folder_id}` ({len(folder_doc_ids)})")
            if not folder_doc_ids:
                st.caption("No document details available for this folder in this Training Set.")
            else:
                for doc_id in folder_doc_ids:
                    item = items_meta.get(doc_id, {})
                    title = item.get("title") or item.get("Brown Title") or item.get("NARA Title") or doc_id
                    summary = item.get("summary", "")
                    doc_date = item.get("date", "N/A")

                    with st.expander(f"📄 {doc_id} — {title}", expanded=False):
                        if summary and str(summary) not in ("", "nan"):
                            st.markdown(f"**Summary:** {summary}")
                        else:
                            st.caption("No summary available for this document.")
                        st.caption(f"**Date:** `{doc_date}` | **Box:** `{item.get('Sushi Box', 'N/A')}`")

    # Tab 3: By Box
    with tabs[3]:
        box_df = u4.compute_box_coverage(parsed_a, folders_meta)
        st.subheader("Box Coverage — Sorted by Coverage % (ascending)")
        st.dataframe(box_df, width="stretch", hide_index=True, height=500)

    # Tab 5: Relevance Coverage — with % for Grade 3 and Grade 1
    with tabs[4]:
        rel_cov = u4.compute_relevance_coverage(parsed_a)
        s = rel_cov["summary"]

        st.markdown(f"""
Across all 45 topics, there are **{s['total_pairs']}** (topic, relevant-folder) pairs.
In this Training Set, **{s['covered_pairs']}** ({s['covered_pairs']/s['total_pairs']*100:.1f}%) have a training document available;
the remaining **{s['uncovered_pairs']}** ({s['uncovered_pairs']/s['total_pairs']*100:.1f}%) can only be found via label match or box/SNC expansion.
        """)

        rc1, rc2 = st.columns(2)
        pct_g3 = round(s['covered_grade3'] / s['total_grade3'] * 100, 1) if s['total_grade3'] > 0 else 0.0
        pct_g1 = round(s['covered_grade1'] / s['total_grade1'] * 100, 1) if s['total_grade1'] > 0 else 0.0
        rc1.metric(
            "⭐⭐⭐ Highly Relevant (Grade 3)",
            f"{s['covered_grade3']} / {s['total_grade3']} ({pct_g3}%)"
        )
        rc2.metric(
            "⭐ Relevant (Grade 1)",
            f"{s['covered_grade1']} / {s['total_grade1']} ({pct_g1}%)"
        )

        st.subheader("Per-Topic Relevance Coverage")
        per_topic = rel_cov["per_topic"]
        st.dataframe(per_topic.sort_values("Coverage %", ascending=True), width="stretch", hide_index=True, height=500)

    # Tab 6: Compare ECFs (if selected)
    if ecf_b and len(tabs) > 5:
        with tabs[5]:
            docs_b = u4.load_ecf_training_docs(ecf_b["path"])
            parsed_b = u4.parse_ecf_docs(docs_b)
            metrics_b = u4.compute_headline_metrics(parsed_b, folders_meta)

            st.subheader(f"Comparison: {ecf_a['label']} vs {ecf_b['label']}")

            # Side-by-side metrics
            comp_col_a, comp_col_b = st.columns(2)
            with comp_col_a:
                st.markdown(f"### {ecf_a['label']}")
                st.metric("Documents", f"{metrics_a['docs_included']:,}", help=f"{metrics_a['docs_pct']}% of total")
                st.metric("Folders Covered", f"{metrics_a['folders_covered']:,}", help=f"{metrics_a['folders_pct']}% covered")
                st.metric("Avg Docs/Folder", f"{metrics_a['avg_docs_per_folder']:.2f}")
            with comp_col_b:
                st.markdown(f"### {ecf_b['label']}")
                st.metric("Documents", f"{metrics_b['docs_included']:,}", help=f"{metrics_b['docs_pct']}% of total")
                st.metric("Folders Covered", f"{metrics_b['folders_covered']:,}", help=f"{metrics_b['folders_pct']}% covered")
                st.metric("Avg Docs/Folder", f"{metrics_b['avg_docs_per_folder']:.2f}")

            lbl_a = ecf_a["label"]
            lbl_b = ecf_b["label"]

            # SNC Coverage Comparison Table
            st.subheader("SNC Coverage Comparison Table")
            snc_a = u4.compute_snc_coverage(parsed_a, folders_meta)
            snc_b = u4.compute_snc_coverage(parsed_b, folders_meta)

            merged_snc = snc_a[["SNC", "Total Folders", "Covered Folders", "Coverage %"]].rename(
                columns={
                    "Covered Folders": f"Covered ({lbl_a})",
                    "Coverage %": f"Coverage % ({lbl_a})",
                }
            )
            merged_snc_b = snc_b[["SNC", "Covered Folders", "Coverage %"]].rename(
                columns={
                    "Covered Folders": f"Covered ({lbl_b})",
                    "Coverage %": f"Coverage % ({lbl_b})",
                }
            )

            comp_df = merged_snc.merge(merged_snc_b, on="SNC", how="outer").fillna(0)
            comp_df["Covered Diff"] = comp_df[f"Covered ({lbl_a})"] - comp_df[f"Covered ({lbl_b})"]

            st.dataframe(
                comp_df.sort_values(f"Coverage % ({lbl_a})", ascending=True),
                width="stretch",
                hide_index=True,
                height=450,
            )

            # Per-Topic Relevance Coverage Comparison Table
            st.subheader("Per-Topic Relevance Coverage Comparison Table")
            rel_cov_a = u4.compute_relevance_coverage(parsed_a)
            rel_cov_b = u4.compute_relevance_coverage(parsed_b)

            df_rel_a = rel_cov_a["per_topic"][["Topic", "Relevant Folders", "Covered", "Coverage %"]].rename(
                columns={"Covered": f"Covered ({lbl_a})", "Coverage %": f"Coverage % ({lbl_a})"}
            )
            df_rel_b = rel_cov_b["per_topic"][["Topic", "Covered", "Coverage %"]].rename(
                columns={"Covered": f"Covered ({lbl_b})", "Coverage %": f"Coverage % ({lbl_b})"}
            )

            topic_comp_df = df_rel_a.merge(df_rel_b, on="Topic", how="outer").fillna(0)
            topic_comp_df["Coverage Diff %"] = (
                topic_comp_df[f"Coverage % ({lbl_a})"] - topic_comp_df[f"Coverage % ({lbl_b})"]
            ).round(1)

            st.dataframe(
                topic_comp_df.sort_values("Coverage Diff %", ascending=False),
                width="stretch",
                hide_index=True,
                height=450,
            )


# ============================================================
def scroll_to_top():
    """Inject fallback JS snippet to scroll the main Streamlit container to the top."""
    st.components.v1.html(
        """
        <script>
            function performScroll() {
                try {
                    const doc = window.parent.document;
                    const targets = [
                        doc.querySelector('section.main'),
                        doc.querySelector('[data-testid="stAppViewContainer"]'),
                        doc.querySelector('.main'),
                        doc.documentElement,
                        doc.body
                    ];
                    targets.forEach(function(el) {
                        if (el) {
                            el.scrollTop = 0;
                            if (typeof el.scrollTo === 'function') {
                                el.scrollTo({top: 0, left: 0, behavior: 'instant'});
                            }
                        }
                    });
                    window.parent.scrollTo({top: 0, left: 0, behavior: 'instant'});
                } catch (e) {
                    console.error('Scroll error:', e);
                }
            }
            performScroll();
            setTimeout(performScroll, 50);
            setTimeout(performScroll, 150);
        </script>
        """,
        height=0,
        width=0,
    )


def main():
    st.sidebar.title("SUSHI BAR")
    app_mode = st.sidebar.radio(
        "Choose Application:",
        [
            "📦 Collection Viewer",
            "🔍 Task Viewer",
            "🧪 Training Set Viewer",
            "🔬 Single Experiment Viewer",
            "⚔️ Two-Experiment Viewer",
            "📖 How-To Guide"
        ]
    )

    if st.session_state.get("_last_app_mode") != app_mode:
        st.session_state["_last_app_mode"] = app_mode
        if scroll_to_here is not None:
            scroll_to_here(delay=0)
        else:
            scroll_to_top()

    if app_mode == "📦 Collection Viewer":
        run_data_overview_ui()
    elif app_mode == "🔍 Task Viewer":
        run_topic_viewer_ui()
    elif app_mode == "🧪 Training Set Viewer":
        run_ecf_inspector_ui()
    elif app_mode == "🔬 Single Experiment Viewer":
        run_single_experiment_ui()
    elif app_mode == "⚔️ Two-Experiment Viewer":
        run_two_experiment_ui()
    elif app_mode == "📖 How-To Guide":
        run_howto_ui()


if __name__ == "__main__":
    main()


