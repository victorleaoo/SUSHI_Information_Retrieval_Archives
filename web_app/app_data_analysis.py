"""
SUSHI Collection — Phase 0: Data Analysis Dashboard
=====================================================
Streamlit application for comprehensive data analysis of the SUSHI
archive collection (U.S. State Department records on Brazil, 1960s-1970s).

Covers all Phase 0 checklist items from experiments.md:
  0.1 SNC Distribution
  0.2 Digitization Coverage per SNC
  0.3 Homophily Feasibility Study
  0.4 Scope Note Coverage
  0.5 Date Range Summary
  0.6 Textual SNC Analysis
  0.7 Document Analysis
"""

import streamlit as st
import json
import os
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SUSHI Phase 0 — Data Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
FOLDERS_PATH = BASE_DIR / "data" / "folders_metadata" / "FoldersV1.3.json"
ITEMS_PATH = BASE_DIR / "data" / "items_metadata" / "itemsV1.2.json"
DISTINCT_SNC_PATH = BASE_DIR / "data" / "folders_metadata" / "distinct_snc.json"
EXPANDED_SNC_PATH = BASE_DIR / "data" / "folders_metadata" / "expanded_snc.json"
ECF_DIR = BASE_DIR / "ecf" / "random_generated"
QRELS_DIR = BASE_DIR / "qrels"

# Color palette
COLORS = {
    "primary": "#6366F1",      # Indigo
    "secondary": "#8B5CF6",    # Purple
    "accent": "#EC4899",       # Pink
    "success": "#10B981",      # Emerald
    "warning": "#F59E0B",      # Amber
    "danger": "#EF4444",       # Red
    "info": "#3B82F6",         # Blue
    "rich": "#10B981",
    "moderate": "#F59E0B",
    "poor": "#EF4444",
    "bg_dark": "#0F172A",
    "bg_card": "#1E293B",
    "text": "#F1F5F9",
}

# ──────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E293B, #334155);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #94A3B8;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #F1F5F9 !important;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #CBD5E1;
    }
    .info-box strong {
        color: #A5B4FC;
    }
    
    .stat-highlight {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        display: block;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Data Loading (cached)
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading folder metadata…")
def load_folders():
    with open(FOLDERS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for fid, meta in data.items():
        row = {"folder_id": fid, **meta}
        rows.append(row)
    df = pd.DataFrame(rows)
    # Parse dates
    df["start_date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["end_date_raw"] = df.get("endDate", "")
    df["end_date"] = pd.to_datetime(df["endDate"], format="%m/%d/%Y", errors="coerce")
    # Extract SNC levels
    df["snc_clean"] = df["snc"].fillna("").astype(str).str.strip()
    df["snc_primary"] = df["snc_clean"].apply(lambda x: x.split()[0] if x else "")
    # 2-level: e.g. "POL 18" from "POL 18-1"
    def get_snc_2level(s):
        parts = s.split()
        if len(parts) < 2:
            return s
        # Remove sub-level: "18-1" -> "18"
        num = parts[1].split("-")[0]
        return f"{parts[0]} {num}"
    df["snc_2level"] = df["snc_clean"].apply(get_snc_2level)
    # 3-level (full SNC)
    df["snc_3level"] = df["snc_clean"]
    # Scope note
    df["has_scope"] = df["raw_scope"].fillna("").str.strip().apply(lambda x: bool(x) and x.lower() != "nan")
    df["scope_text"] = df["raw_scope"].fillna("").str.strip()
    # Start year
    df["start_year"] = df["start_date"].dt.year
    return df


@st.cache_data(show_spinner="Loading document metadata (this may take a moment)…")
def load_items():
    with open(ITEMS_PATH, "r", encoding="utf-8") as f:
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


@st.cache_data(show_spinner="Loading distinct SNC data…")
def load_distinct_snc():
    with open(DISTINCT_SNC_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_data(show_spinner="Loading ECF…")
def load_ecf(ecf_path):
    """Load an ECF file and return a set of (folder_id) that have training docs."""
    with open(ecf_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Parse training documents: "BOX/FOLDER/FILE.pdf"
    folder_doc_map = defaultdict(list)
    for exp_set in data.get("ExperimentSets", []):
        for doc_path in exp_set.get("TrainingDocuments", []):
            parts = doc_path.split("/")
            if len(parts) >= 3:
                folder_id = parts[1]
                doc_file = parts[2].replace(".pdf", "")
                folder_doc_map[folder_id].append(doc_file)
    return dict(folder_doc_map)


@st.cache_data(show_spinner="Loading full items JSON for document detail…")
def load_items_raw():
    """Load full items JSON. Used only when drilling into specific documents."""
    with open(ITEMS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_ecf_files(ecf_type="uniform"):
    """Get list of ECF files of a given type."""
    files = []
    for f in sorted(ECF_DIR.iterdir()):
        if ecf_type == "uniform" and f.name.startswith("ECF_RANDOM_"):
            files.append(f)
        elif ecf_type == "skewed" and f.name.startswith("ECF_UNEVEN_"):
            files.append(f)
    return files


# ──────────────────────────────────────────────────────────────────────
# Helper: build folder<->document mapping
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building folder-document index…")
def build_folder_doc_index(_items_df):
    """Map folder_id -> list of doc_ids, and compute doc counts per folder."""
    folder_docs = _items_df.groupby("folder_id")["doc_id"].apply(list).to_dict()
    folder_doc_counts = _items_df.groupby("folder_id")["doc_id"].count().to_dict()
    return folder_docs, folder_doc_counts


# ──────────────────────────────────────────────────────────────────────
# Helper: common words from titles/summaries
# ──────────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "we", "our", "my", "your", "not", "no", "nor",
    "also", "than", "other", "which", "who", "whom", "what", "when",
    "where", "how", "all", "each", "every", "both", "few", "more", "most",
    "some", "any", "such", "only", "own", "same", "so", "very", "just",
    "about", "up", "out", "into", "over", "after", "before", "between",
    "under", "above", "below", "during", "through", "while", "against",
    "documents", "document", "including", "new", "united", "states",
    "government", "brazil", "brazilian", "u.s.", "u.s", "state",
    "department", "embassy", "president", "regarding", "report",
    "according", "one", "two", "three", "four", "five", "among",
    "well", "however", "several", "various", "includes", "related",
    "discussed", "described", "reported", "mentioned", "involved",
    "provided", "based", "within", "1964", "1965", "1966", "1967",
    "1968", "1969", "1970", "1971", "1972", "1973", "1974",
}


def extract_keywords(texts, top_n=25):
    """Extract top keywords from a list of texts, excluding stop words."""
    word_counter = Counter()
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        word_counter.update(words)
    return word_counter.most_common(top_n)


# ──────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ──────────────────────────────────────────────────────────────────────
def apply_dark_theme(fig, height=500):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#CBD5E1"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.1)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.1)")
    return fig


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 SUSHI Explorer")
    st.markdown("**Phase 0: Data Analysis**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 0.1 SNC Distribution",
            "🔍 0.2 Digitization Coverage",
            "🤝 0.3 Homophily Feasibility",
            "📝 0.4 Scope Notes",
            "📅 0.5 Date Ranges",
            "🏷️ 0.6 Textual SNC",
            "📄 0.7 Documents",
            "🔎 SNC Deep Dive",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:#64748B;font-size:0.75rem;">'
        "SUSHI Collection · Victor Hugo Oliveira Leão, 2026"
        "</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────
folders_df = load_folders()
items_df = load_items()
folder_docs, folder_doc_counts = build_folder_doc_index(items_df)

# Enrich folders with doc counts
folders_df["doc_count"] = folders_df["folder_id"].map(folder_doc_counts).fillna(0).astype(int)


# ══════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 📦 SUSHI Collection — Phase 0 Data Analysis")
    st.markdown(
        '<div class="info-box">'
        "<strong>Purpose:</strong> Comprehensive data analysis of the SUSHI archive collection "
        "to inform decisions for Phases 1-3 of the experiment pipeline. "
        "This dashboard covers all Phase 0 checklist items from <code>experiments.md</code>."
        "</div>",
        unsafe_allow_html=True,
    )

    # Key metrics
    n_folders = len(folders_df)
    n_docs = len(items_df)
    n_boxes = folders_df["box"].nunique()
    n_snc_distinct = folders_df["snc_3level"].nunique()
    n_snc_primary = folders_df["snc_primary"].nunique()
    n_with_scope = folders_df["has_scope"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("📁 Total Folders", f"{n_folders:,}")
    col2.metric("📄 Total Documents", f"{n_docs:,}")
    col3.metric("📦 Total Boxes", f"{n_boxes:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("🏷️ Distinct SNCs (3-level)", f"{n_snc_distinct}")
    col5.metric("🏷️ Primary SNC Codes", f"{n_snc_primary}")
    col6.metric("📝 Folders with Scope Notes", f"{n_with_scope}")

    st.markdown("---")

    # Docs per folder distribution
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Documents per Folder Distribution")
        fig = px.histogram(
            folders_df, x="doc_count", nbins=50,
            color_discrete_sequence=[COLORS["primary"]],
            labels={"doc_count": "Number of Documents", "count": "Frequency"},
        )
        fig = apply_dark_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Folders per Box Distribution")
        folders_per_box = folders_df.groupby("box").size().reset_index(name="folder_count")
        fig = px.histogram(
            folders_per_box, x="folder_count", nbins=30,
            color_discrete_sequence=[COLORS["secondary"]],
            labels={"folder_count": "Number of Folders", "count": "Frequency"},
        )
        fig = apply_dark_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Primary SNC overview
    st.markdown("### Primary SNC Code — Folder & Document Counts")
    primary_stats = (
        folders_df.groupby("snc_primary")
        .agg(folder_count=("folder_id", "count"), total_docs=("doc_count", "sum"))
        .reset_index()
        .sort_values("folder_count", ascending=False)
    )
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Folders per Primary SNC", "Documents per Primary SNC"),
        shared_yaxes=True,
    )
    fig.add_trace(
        go.Bar(
            y=primary_stats["snc_primary"],
            x=primary_stats["folder_count"],
            orientation="h",
            marker_color=COLORS["primary"],
            name="Folders",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            y=primary_stats["snc_primary"],
            x=primary_stats["total_docs"],
            orientation="h",
            marker_color=COLORS["accent"],
            name="Documents",
        ),
        row=1, col=2,
    )
    fig = apply_dark_theme(fig, height=max(400, len(primary_stats) * 22))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: 0.1 SNC Distribution
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 0.1 SNC Distribution":
    st.markdown("# 📊 0.1 SNC Distribution")
    st.markdown(
        '<div class="info-box">'
        "<strong>Checklist:</strong> Count folders per 3-level, 2-level, and 1-level SNC. "
        "Plot histogram. List top-10 and bottom-10 SNCs. Flag SNCs with scope notes (F3/F4)."
        "</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "3-Level SNC (Full)", "2-Level SNC", "1-Level (Primary)", "Documents per SNC"
    ])

    # ── Tab 1: 3-Level ──
    with tab1:
        snc3 = (
            folders_df.groupby("snc_3level")
            .agg(
                folder_count=("folder_id", "count"),
                total_docs=("doc_count", "sum"),
                has_scope=("has_scope", "any"),
            )
            .reset_index()
            .sort_values("folder_count", ascending=False)
        )

        st.metric("Distinct 3-Level SNCs", len(snc3))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top-10 SNCs by Folder Count")
            top10 = snc3.head(10).copy()
            top10["scope?"] = top10["has_scope"].map({True: "✅", False: "❌"})
            st.dataframe(
                top10[["snc_3level", "folder_count", "total_docs", "scope?"]],
                use_container_width=True, hide_index=True,
            )

        with col2:
            st.markdown("#### Bottom-10 SNCs by Folder Count")
            bottom10 = snc3.tail(10).copy()
            bottom10["scope?"] = bottom10["has_scope"].map({True: "✅", False: "❌"})
            st.dataframe(
                bottom10[["snc_3level", "folder_count", "total_docs", "scope?"]],
                use_container_width=True, hide_index=True,
            )

        st.markdown("#### Histogram: Folders per SNC (3-Level)")
        fig = px.histogram(
            snc3, x="folder_count", nbins=40,
            color_discrete_sequence=[COLORS["primary"]],
            labels={"folder_count": "Number of Folders", "count": "Number of SNCs"},
        )
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### All SNCs — Complete Table")
        st.dataframe(snc3.rename(columns={
            "snc_3level": "SNC", "folder_count": "Folders",
            "total_docs": "Documents", "has_scope": "Has Scope Note",
        }), use_container_width=True, hide_index=True, height=400)

    # ── Tab 2: 2-Level ──
    with tab2:
        snc2 = (
            folders_df.groupby("snc_2level")
            .agg(folder_count=("folder_id", "count"), total_docs=("doc_count", "sum"))
            .reset_index()
            .sort_values("folder_count", ascending=False)
        )
        st.metric("Distinct 2-Level SNCs", len(snc2))

        fig = px.bar(
            snc2.head(40), x="snc_2level", y="folder_count",
            color="total_docs",
            color_continuous_scale="Viridis",
            labels={"snc_2level": "SNC (2-Level)", "folder_count": "Folders", "total_docs": "Documents"},
            title="Top 40 SNCs (2-Level) by Folder Count",
        )
        fig = apply_dark_theme(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(snc2.rename(columns={
            "snc_2level": "SNC (2-Level)", "folder_count": "Folders", "total_docs": "Documents",
        }), use_container_width=True, hide_index=True, height=400)

    # ── Tab 3: 1-Level (Primary) ──
    with tab3:
        snc1 = (
            folders_df.groupby("snc_primary")
            .agg(folder_count=("folder_id", "count"), total_docs=("doc_count", "sum"))
            .reset_index()
            .sort_values("folder_count", ascending=False)
        )
        st.metric("Distinct Primary SNC Codes", len(snc1))

        fig = px.bar(
            snc1, x="snc_primary", y="folder_count",
            color="total_docs",
            color_continuous_scale="Plasma",
            labels={"snc_primary": "Primary SNC", "folder_count": "Folders", "total_docs": "Documents"},
            title="Folders per Primary SNC Code",
        )
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.treemap(
            snc1, path=["snc_primary"], values="folder_count",
            color="total_docs", color_continuous_scale="Viridis",
            title="Treemap: Primary SNC by Folder Count",
        )
        fig2 = apply_dark_theme(fig2, height=500)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4: Documents per SNC ──
    with tab4:
        st.markdown("#### Document Distribution across SNC Codes")
        snc_docs = (
            folders_df.groupby("snc_3level")
            .agg(total_docs=("doc_count", "sum"), folder_count=("folder_id", "count"))
            .reset_index()
            .sort_values("total_docs", ascending=False)
        )
        snc_docs["avg_docs_per_folder"] = (snc_docs["total_docs"] / snc_docs["folder_count"]).round(1)

        fig = px.scatter(
            snc_docs, x="folder_count", y="total_docs",
            hover_data=["snc_3level", "avg_docs_per_folder"],
            color="avg_docs_per_folder",
            color_continuous_scale="Turbo",
            labels={
                "folder_count": "Number of Folders",
                "total_docs": "Total Documents",
                "avg_docs_per_folder": "Avg Docs/Folder",
            },
            title="SNC: Folder Count vs Document Count",
        )
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top 20 SNCs by Total Documents")
        st.dataframe(
            snc_docs.head(20).rename(columns={
                "snc_3level": "SNC", "total_docs": "Documents",
                "folder_count": "Folders", "avg_docs_per_folder": "Avg Docs/Folder",
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE: 0.2 Digitization Coverage
# ══════════════════════════════════════════════════════════════════════
elif page == "🔍 0.2 Digitization Coverage":
    st.markdown("# 🔍 0.2 Digitization Coverage per SNC")
    st.markdown(
        '<div class="info-box">'
        "<strong>Checklist:</strong> For each SNC, compute (folders with ≥1 training doc) / (total folders). "
        "Classify: <strong style='color:#10B981'>Rich ≥50%</strong>, "
        "<strong style='color:#F59E0B'>Moderate 20-49%</strong>, "
        "<strong style='color:#EF4444'>Poor &lt;20%</strong>. "
        "Identify SNCs where &gt;80% folders are empty."
        "</div>",
        unsafe_allow_html=True,
    )

    # ECF selection
    ecf_files = get_ecf_files("uniform")
    ecf_names = [f.stem for f in ecf_files]
    selected_ecf_name = st.selectbox("Select Uniform ECF", ecf_names, index=0)
    selected_ecf_path = ECF_DIR / f"{selected_ecf_name}.json"

    ecf_data = load_ecf(str(selected_ecf_path))

    # Compute coverage per SNC
    folders_df["has_training_doc"] = folders_df["folder_id"].isin(ecf_data.keys())

    coverage = (
        folders_df.groupby("snc_3level")
        .agg(
            total_folders=("folder_id", "count"),
            folders_with_docs=("has_training_doc", "sum"),
            total_all_docs=("doc_count", "sum"),
        )
        .reset_index()
    )
    coverage["coverage_pct"] = (coverage["folders_with_docs"] / coverage["total_folders"] * 100).round(1)
    coverage["category"] = coverage["coverage_pct"].apply(
        lambda x: "Rich (≥50%)" if x >= 50 else ("Moderate (20-49%)" if x >= 20 else "Poor (<20%)")
    )
    coverage = coverage.sort_values("coverage_pct", ascending=False)

    # Category counts
    cat_counts = coverage["category"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Rich SNCs", cat_counts.get("Rich (≥50%)", 0))
    col2.metric("🟡 Moderate SNCs", cat_counts.get("Moderate (20-49%)", 0))
    col3.metric("🔴 Poor SNCs", cat_counts.get("Poor (<20%)", 0))
    col4.metric("Total Folders w/ Training Docs", sum(ecf_data.keys().__contains__(fid) for fid in folders_df["folder_id"]))

    # Category distribution pie
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(
            coverage, names="category", title="Digitization Category Distribution",
            color="category",
            color_discrete_map={
                "Rich (≥50%)": COLORS["rich"],
                "Moderate (20-49%)": COLORS["moderate"],
                "Poor (<20%)": COLORS["poor"],
            },
        )
        fig = apply_dark_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.histogram(
            coverage, x="coverage_pct", nbins=20,
            color="category",
            color_discrete_map={
                "Rich (≥50%)": COLORS["rich"],
                "Moderate (20-49%)": COLORS["moderate"],
                "Poor (<20%)": COLORS["poor"],
            },
            labels={"coverage_pct": "Coverage %", "count": "Number of SNCs"},
            title="Digitization Coverage Distribution",
        )
        fig = apply_dark_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # SNCs where >80% are empty
    st.markdown("### ⚠️ SNCs with >80% Empty Folders (Homophily won't help)")
    empty_sncs = coverage[coverage["coverage_pct"] < 20].sort_values("total_folders", ascending=False)
    st.dataframe(
        empty_sncs.rename(columns={
            "snc_3level": "SNC", "total_folders": "Total Folders",
            "folders_with_docs": "Folders w/ Docs", "coverage_pct": "Coverage %",
            "category": "Category",
        }),
        use_container_width=True, hide_index=True, height=300,
    )

    # Full table
    with st.expander("📋 Full Coverage Table"):
        st.dataframe(
            coverage.rename(columns={
                "snc_3level": "SNC", "total_folders": "Total Folders",
                "folders_with_docs": "Folders w/ Docs", "coverage_pct": "Coverage %",
                "total_all_docs": "Total Collection Docs", "category": "Category",
            }),
            use_container_width=True, hide_index=True, height=500,
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE: 0.3 Homophily Feasibility
# ══════════════════════════════════════════════════════════════════════
elif page == "🤝 0.3 Homophily Feasibility":
    st.markdown("# 🤝 0.3 Homophily Feasibility Study")
    st.markdown(
        '<div class="info-box">'
        "<strong>Checklist:</strong> For empty folders in a Uniform ECF, count available same-SNC neighbors "
        "with training docs. Compute % of empty folders with ≥3 and ≥1 same-SNC docs."
        "</div>",
        unsafe_allow_html=True,
    )

    ecf_files = get_ecf_files("uniform")
    ecf_names = [f.stem for f in ecf_files]
    selected_ecf_name = st.selectbox("Select Uniform ECF", ecf_names, index=0, key="homo_ecf")
    selected_ecf_path = ECF_DIR / f"{selected_ecf_name}.json"
    ecf_data = load_ecf(str(selected_ecf_path))

    # Threshold slider
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        threshold = st.slider("Minimum neighbor docs threshold (D7)", 1, 10, 3)
    with col_t2:
        match_level = st.radio("SNC matching level (D6)", ["Exact SNC", "2-Level Prefix", "Both"])

    # Compute homophily
    folders_with_docs_set = set(ecf_data.keys())
    empty_folders = folders_df[~folders_df["folder_id"].isin(folders_with_docs_set)].copy()

    # Build SNC->folders with docs mapping
    folders_with_docs_df = folders_df[folders_df["folder_id"].isin(folders_with_docs_set)]

    snc_exact_docs = defaultdict(list)
    snc_2level_docs = defaultdict(list)
    box_docs = defaultdict(list)

    for _, row in folders_with_docs_df.iterrows():
        fid = row["folder_id"]
        doc_ids = ecf_data.get(fid, [])
        snc_exact_docs[row["snc_3level"]].extend(doc_ids)
        snc_2level_docs[row["snc_2level"]].extend(doc_ids)
        box_docs[row["box"]].extend(doc_ids)

    # For each empty folder, count available neighbor docs
    results = []
    for _, row in empty_folders.iterrows():
        exact_count = len(snc_exact_docs.get(row["snc_3level"], []))
        prefix_count = len(snc_2level_docs.get(row["snc_2level"], []))
        box_count = len(box_docs.get(row["box"], []))

        if match_level == "Exact SNC":
            neighbor_docs = exact_count
        elif match_level == "2-Level Prefix":
            neighbor_docs = prefix_count
        else:  # Both
            neighbor_docs = max(exact_count, prefix_count)

        results.append({
            "folder_id": row["folder_id"],
            "snc": row["snc_3level"],
            "box": row["box"],
            "exact_snc_docs": exact_count,
            "prefix_snc_docs": prefix_count,
            "same_box_docs": box_count,
            "neighbor_docs": neighbor_docs,
        })

    homo_df = pd.DataFrame(results)

    # Key stats
    n_empty = len(homo_df)
    n_above_thresh = (homo_df["neighbor_docs"] >= threshold).sum()
    n_above_1 = (homo_df["neighbor_docs"] >= 1).sum()
    pct_above_thresh = (n_above_thresh / n_empty * 100) if n_empty > 0 else 0
    pct_above_1 = (n_above_1 / n_empty * 100) if n_empty > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Empty Folders", f"{n_empty:,}")
    col2.metric(f"≥{threshold} Neighbor Docs", f"{n_above_thresh:,}", f"{pct_above_thresh:.1f}%")
    col3.metric("≥1 Neighbor Docs", f"{n_above_1:,}", f"{pct_above_1:.1f}%")
    col4.metric("Homophily Viability", "✅ Yes" if pct_above_thresh >= 20 else "❌ No")

    st.markdown(
        f'<div class="info-box">'
        f"<strong>Decision D9:</strong> {'Proceed with Homophily' if pct_above_thresh >= 20 else 'Homophily NOT viable'} — "
        f"{pct_above_thresh:.1f}% of empty folders have ≥{threshold} same-SNC neighbor docs "
        f"(threshold for viability: ≥20%)."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Distribution of neighbor docs
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            homo_df, x="neighbor_docs", nbins=50,
            color_discrete_sequence=[COLORS["primary"]],
            labels={"neighbor_docs": "# Same-SNC Neighbor Docs", "count": "# Empty Folders"},
            title="Distribution: Neighbor Docs Available per Empty Folder",
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text=f"Threshold={threshold}")
        fig = apply_dark_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Cumulative distribution
        sorted_counts = sorted(homo_df["neighbor_docs"].values)
        cumulative = [(i + 1) / len(sorted_counts) * 100 for i in range(len(sorted_counts))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_counts, y=cumulative,
            mode="lines", fill="tozeroy",
            line=dict(color=COLORS["accent"], width=2),
            fillcolor="rgba(236, 72, 153, 0.2)",
        ))
        fig.update_layout(
            title="Cumulative Distribution of Neighbor Docs",
            xaxis_title="# Neighbor Docs",
            yaxis_title="Cumulative % of Empty Folders",
        )
        fig.add_hline(y=80, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text="80%")
        fig = apply_dark_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Breakdown by SNC
    with st.expander("📋 Per-SNC Homophily Breakdown"):
        snc_homo = (
            homo_df.groupby("snc")
            .agg(
                empty_folders=("folder_id", "count"),
                avg_neighbor_docs=("neighbor_docs", "mean"),
                max_neighbor_docs=("neighbor_docs", "max"),
                pct_above_threshold=("neighbor_docs", lambda x: (x >= threshold).mean() * 100),
            )
            .reset_index()
            .sort_values("pct_above_threshold", ascending=False)
        )
        snc_homo["avg_neighbor_docs"] = snc_homo["avg_neighbor_docs"].round(1)
        snc_homo["pct_above_threshold"] = snc_homo["pct_above_threshold"].round(1)
        st.dataframe(snc_homo, use_container_width=True, hide_index=True, height=400)


# ══════════════════════════════════════════════════════════════════════
# PAGE: 0.4 Scope Notes
# ══════════════════════════════════════════════════════════════════════
elif page == "📝 0.4 Scope Notes":
    st.markdown("# 📝 0.4 Scope Note Coverage")
    st.markdown(
        '<div class="info-box">'
        "<strong>Checklist:</strong> Count folders with non-empty scope notes. "
        "List which primary SNC codes have scope notes (predicts F3/F4 benefit)."
        "</div>",
        unsafe_allow_html=True,
    )

    n_with_scope = folders_df["has_scope"].sum()
    n_total = len(folders_df)
    pct_scope = n_with_scope / n_total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Folders WITH Scope Notes", f"{n_with_scope}")
    col2.metric("Folders WITHOUT Scope Notes", f"{n_total - n_with_scope}")
    col3.metric("Coverage", f"{pct_scope:.1f}%")

    # SNCs with scope
    snc_scope = (
        folders_df.groupby("snc_3level")
        .agg(
            folder_count=("folder_id", "count"),
            folders_with_scope=("has_scope", "sum"),
        )
        .reset_index()
    )
    snc_scope["scope_pct"] = (snc_scope["folders_with_scope"] / snc_scope["folder_count"] * 100).round(1)

    snc_with_scope = snc_scope[snc_scope["folders_with_scope"] > 0].sort_values("folders_with_scope", ascending=False)
    snc_without_scope = snc_scope[snc_scope["folders_with_scope"] == 0]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("SNCs WITH Scope Notes", len(snc_with_scope))
        st.markdown("**Note:** Only these SNCs benefit from F3/F4 index configurations.")
    with col_b:
        st.metric("SNCs WITHOUT Scope Notes", len(snc_without_scope))

    # Primary SNC breakdown
    st.markdown("### Primary SNC Codes with Scope Notes")
    primary_scope = (
        folders_df.groupby("snc_primary")
        .agg(
            total_folders=("folder_id", "count"),
            with_scope=("has_scope", "sum"),
        )
        .reset_index()
    )
    primary_scope["pct"] = (primary_scope["with_scope"] / primary_scope["total_folders"] * 100).round(1)
    primary_scope = primary_scope.sort_values("with_scope", ascending=False)

    fig = px.bar(
        primary_scope, x="snc_primary", y=["with_scope", "total_folders"],
        barmode="group",
        color_discrete_sequence=[COLORS["success"], COLORS["primary"]],
        labels={"value": "Count", "snc_primary": "Primary SNC", "variable": ""},
        title="Scope Note Coverage by Primary SNC",
    )
    fig = apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Show actual scope notes
    with st.expander("📖 View All Scope Notes"):
        scope_notes = folders_df[folders_df["has_scope"]][["snc_3level", "label", "scope_text"]].drop_duplicates(
            subset=["snc_3level"]
        ).sort_values("snc_3level")
        for _, row in scope_notes.iterrows():
            st.markdown(f"**{row['snc_3level']}:** {row['scope_text']}")
            st.markdown("---")


# ══════════════════════════════════════════════════════════════════════
# PAGE: 0.5 Date Ranges
# ══════════════════════════════════════════════════════════════════════
elif page == "📅 0.5 Date Ranges":
    st.markdown("# 📅 0.5 Date Range Summary")
    st.markdown(
        '<div class="info-box">'
        "<strong>Checklist:</strong> Compute distribution of folder date ranges (start year). "
        "Flag folders with missing or anomalous dates."
        "</div>",
        unsafe_allow_html=True,
    )

    valid_dates = folders_df[folders_df["start_date"].notna()]
    missing_start = folders_df["start_date"].isna().sum()
    missing_end = folders_df["end_date"].isna().sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valid Start Dates", f"{len(valid_dates)}")
    col2.metric("Missing Start Dates", f"{missing_start}")
    col3.metric("Missing End Dates", f"{missing_end}")
    col4.metric("Date Range", f"{valid_dates['start_year'].min():.0f} – {valid_dates['start_year'].max():.0f}" if len(valid_dates) > 0 else "N/A")

    # Year distribution
    fig = px.histogram(
        valid_dates, x="start_year", nbins=20,
        color_discrete_sequence=[COLORS["primary"]],
        labels={"start_year": "Start Year", "count": "Number of Folders"},
        title="Distribution of Folder Start Years",
    )
    fig = apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Year by primary SNC
    st.markdown("### Start Year Distribution by Primary SNC")
    year_snc = valid_dates.groupby(["start_year", "snc_primary"]).size().reset_index(name="count")
    fig = px.bar(
        year_snc, x="start_year", y="count", color="snc_primary",
        labels={"start_year": "Start Year", "count": "Folders", "snc_primary": "Primary SNC"},
        title="Folders by Year and Primary SNC",
    )
    fig = apply_dark_theme(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Anomalous dates
    st.markdown("### ⚠️ Folders with Missing or Anomalous Dates")
    anomalous = folders_df[
        (folders_df["start_date"].isna()) |
        (folders_df["end_date_raw"].str.contains("Unknown", case=False, na=False))
    ][["folder_id", "box", "snc_3level", "label", "date", "endDate"]].copy()

    if len(anomalous) > 0:
        st.warning(f"Found {len(anomalous)} folders with missing or anomalous dates")
        st.dataframe(anomalous, use_container_width=True, hide_index=True, height=300)
    else:
        st.success("No anomalous dates found!")


# ══════════════════════════════════════════════════════════════════════
# PAGE: 0.6 Textual SNC
# ══════════════════════════════════════════════════════════════════════
elif page == "🏷️ 0.6 Textual SNC":
    st.markdown("# 🏷️ 0.6 Textual SNC Analysis")
    st.markdown(
        '<div class="info-box">'
        "<strong>Checklist:</strong> Check which SNC values have null/missing parent labels, "
        "scope notes, etc. Analyze the textual content of SNC codes."
        "</div>",
        unsafe_allow_html=True,
    )

    # Check for null/missing SNC values
    null_snc = folders_df[
        (folders_df["snc_clean"] == "") |
        (folders_df["snc_clean"].str.lower() == "nan") |
        (folders_df["snc_clean"].isna())
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Folders with NULL/Missing SNC", len(null_snc))
    with col2:
        null_parent = folders_df[
            (folders_df["parent"].isna()) |
            (folders_df["parent"].astype(str).str.lower() == "none") |
            (folders_df["parent"].astype(str).str.lower() == "nan")
        ]
        st.metric("Folders with NULL/Missing Parent SNC", len(null_parent))

    if len(null_snc) > 0:
        st.warning(f"⚠️ {len(null_snc)} folders have missing SNC codes!")
        st.dataframe(null_snc[["folder_id", "box", "label"]], use_container_width=True, hide_index=True)

    # SNC label analysis
    st.markdown("### SNC Label Quality Analysis")

    snc_labels = folders_df.groupby("snc_3level").agg(
        parent_code=("parent", "first"),
        label_parent_expanded=("label_parent_expanded", "first"),
        has_scope=("has_scope", "any"),
        folder_count=("folder_id", "count"),
        label1965=("label1965", "first"),
        label1963=("label1963", "first"),
    ).reset_index()

    snc_labels["has_1965_label"] = snc_labels["label1965"].apply(
        lambda x: bool(x) and str(x).lower() != "nan"
    )
    snc_labels["has_1963_label"] = snc_labels["label1963"].apply(
        lambda x: bool(x) and str(x).lower() != "nan"
    )
    snc_labels["is_top_level"] = snc_labels["parent_code"].apply(
        lambda x: str(x).lower() in ("none", "nan", "")
    )

    tab1, tab2, tab3 = st.tabs(["Label Completeness", "Top-Level SNCs", "Label Text Analysis"])

    with tab1:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("SNCs with 1965 Label", snc_labels["has_1965_label"].sum())
        col_b.metric("SNCs with 1963 Label", snc_labels["has_1963_label"].sum())
        col_c.metric("Top-Level SNCs (no parent)", snc_labels["is_top_level"].sum())

        missing_1965 = snc_labels[~snc_labels["has_1965_label"]]
        if len(missing_1965) > 0:
            st.markdown("#### SNCs Missing 1965 Label")
            st.dataframe(missing_1965[["snc_3level", "label_parent_expanded", "folder_count"]],
                         use_container_width=True, hide_index=True)

    with tab2:
        top_level = snc_labels[snc_labels["is_top_level"]].sort_values("folder_count", ascending=False)
        st.markdown("These are top-level SNC codes (no parent hierarchy):")
        fig = px.bar(
            top_level, x="snc_3level", y="folder_count",
            color="has_scope",
            color_discrete_map={True: COLORS["success"], False: COLORS["danger"]},
            labels={"snc_3level": "SNC", "folder_count": "Folders", "has_scope": "Has Scope"},
            title="Top-Level SNC Codes",
        )
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Expanded SNC Label Word Frequency")
        all_labels = folders_df["label_parent_expanded"].dropna().tolist()
        keywords = extract_keywords(all_labels, top_n=30)
        if keywords:
            kw_df = pd.DataFrame(keywords, columns=["Word", "Frequency"])
            fig = px.bar(
                kw_df, x="Frequency", y="Word", orientation="h",
                color="Frequency", color_continuous_scale="Viridis",
                title="Most Common Words in Expanded SNC Labels",
            )
            fig = apply_dark_theme(fig, height=600)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: 0.7 Documents
# ══════════════════════════════════════════════════════════════════════
elif page == "📄 0.7 Documents":
    st.markdown("# 📄 0.7 Document Analysis")
    st.markdown(
        '<div class="info-box">'
        "<strong>Checklist:</strong> Analyze document Titles, Summaries, and OCR to check "
        "if they make sense within their SNC folders."
        "</div>",
        unsafe_allow_html=True,
    )

    # Overview stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", f"{len(items_df):,}")
    col2.metric("With Titles", f"{items_df['has_title'].sum():,}")
    col3.metric("With Summaries", f"{items_df['has_summary'].sum():,}")
    col4.metric("Avg OCR Pages", f"{items_df['ocr_pages'].mean():.1f}")

    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🔍 Browse by SNC", "📑 Search Documents"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                items_df, x="ocr_pages", nbins=50,
                color_discrete_sequence=[COLORS["primary"]],
                labels={"ocr_pages": "Number of OCR Pages", "count": "Documents"},
                title="Distribution of OCR Pages per Document",
            )
            fig = apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Documents per folder distribution
            docs_per_folder = items_df.groupby("folder_id").size().reset_index(name="doc_count")
            fig = px.histogram(
                docs_per_folder, x="doc_count", nbins=50,
                color_discrete_sequence=[COLORS["accent"]],
                labels={"doc_count": "Documents per Folder", "count": "Folders"},
                title="Distribution of Documents per Folder",
            )
            fig = apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Document titles — keyword analysis
        st.markdown("### Most Common Words in Document Titles")
        title_keywords = extract_keywords(items_df["title"].dropna().tolist(), top_n=30)
        if title_keywords:
            kw_df = pd.DataFrame(title_keywords, columns=["Word", "Count"])
            fig = px.bar(
                kw_df, x="Count", y="Word", orientation="h",
                color="Count", color_continuous_scale="Plasma",
                title="Top 30 Words in Document Titles (stopwords removed)",
            )
            fig = apply_dark_theme(fig, height=600)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Select SNC to browse
        snc_options = sorted(folders_df["snc_3level"].unique().tolist())
        selected_snc = st.selectbox("Select SNC Code", snc_options, index=0)

        snc_folders = folders_df[folders_df["snc_3level"] == selected_snc]
        snc_folder_ids = set(snc_folders["folder_id"].tolist())
        snc_docs = items_df[items_df["folder_id"].isin(snc_folder_ids)]

        col1, col2, col3 = st.columns(3)
        col1.metric("Folders in SNC", len(snc_folders))
        col2.metric("Documents in SNC", len(snc_docs))
        col3.metric(
            "Expanded Label",
            snc_folders["label_parent_expanded"].iloc[0] if len(snc_folders) > 0 else "N/A",
        )

        # Keywords for this SNC
        st.markdown(f"### Top Words in '{selected_snc}' Document Titles & Summaries")
        snc_texts = snc_docs["title"].dropna().tolist() + snc_docs["summary"].dropna().tolist()
        snc_keywords = extract_keywords(snc_texts, top_n=20)
        if snc_keywords:
            kw_df = pd.DataFrame(snc_keywords, columns=["Word", "Count"])
            fig = px.bar(
                kw_df, x="Count", y="Word", orientation="h",
                color="Count", color_continuous_scale="Turbo",
            )
            fig = apply_dark_theme(fig, height=450)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        # Show folders and documents
        st.markdown("### Folders")
        for _, folder_row in snc_folders.iterrows():
            fid = folder_row["folder_id"]
            folder_label = folder_row["label"]
            n_docs = folder_row["doc_count"]
            with st.expander(f"📁 {folder_label} ({n_docs} docs) — ID: {fid}"):
                folder_items = snc_docs[snc_docs["folder_id"] == fid]
                if len(folder_items) == 0:
                    st.info("No documents in this folder.")
                else:
                    for _, doc_row in folder_items.head(20).iterrows():
                        st.markdown(f"**📄 {doc_row['doc_id']}** — {doc_row['title']}")
                        if doc_row["has_summary"]:
                            st.caption(doc_row.get("summary", "")[:300] + "…" if len(str(doc_row.get("summary", ""))) > 300 else doc_row.get("summary", ""))
                        st.markdown("---")
                    if len(folder_items) > 20:
                        st.info(f"Showing first 20 of {len(folder_items)} documents.")

    with tab3:
        search_query = st.text_input("🔍 Search documents by title", placeholder="Type a keyword…")
        if search_query:
            mask = items_df["title"].fillna("").str.contains(search_query, case=False, na=False)
            results = items_df[mask].head(50)
            st.info(f"Found {mask.sum()} documents matching '{search_query}' (showing first 50)")

            # Enrich with folder info
            results_enriched = results.merge(
                folders_df[["folder_id", "snc_3level", "label", "label_parent_expanded"]],
                on="folder_id", how="left",
            )
            st.dataframe(
                results_enriched[["doc_id", "title", "snc_3level", "label", "date"]],
                use_container_width=True, hide_index=True, height=400,
            )


# ══════════════════════════════════════════════════════════════════════
# PAGE: SNC Deep Dive
# ══════════════════════════════════════════════════════════════════════
elif page == "🔎 SNC Deep Dive":
    st.markdown("# 🔎 SNC Deep Dive")
    st.markdown(
        '<div class="info-box">'
        "Select any SNC code for a comprehensive view: folder list, documents, "
        "keyword analysis, scope notes, date range, and digitization status."
        "</div>",
        unsafe_allow_html=True,
    )

    # SNC selector with search
    snc_options = sorted(folders_df["snc_3level"].unique().tolist())
    
    # Add label info to the selector
    snc_labels_map = (
        folders_df.groupby("snc_3level")["label_parent_expanded"]
        .first()
        .to_dict()
    )
    snc_display = [f"{s} — {snc_labels_map.get(s, '')}" for s in snc_options]
    
    selected_idx = st.selectbox(
        "Select SNC Code",
        range(len(snc_options)),
        format_func=lambda i: snc_display[i],
    )
    selected_snc = snc_options[selected_idx]

    # Get data for this SNC
    snc_folders = folders_df[folders_df["snc_3level"] == selected_snc].copy()
    snc_folder_ids = set(snc_folders["folder_id"].tolist())
    snc_docs = items_df[items_df["folder_id"].isin(snc_folder_ids)]

    # Header metrics
    st.markdown(f"## {selected_snc}")
    expanded = snc_folders["label_parent_expanded"].iloc[0] if len(snc_folders) > 0 else "N/A"
    st.markdown(f"**Expanded:** {expanded}")

    scope_text = snc_folders[snc_folders["has_scope"]]["scope_text"].iloc[0] if snc_folders["has_scope"].any() else None
    if scope_text:
        st.markdown(f"**Scope Note:** {scope_text}")
    else:
        st.info("No scope note available for this SNC.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📁 Folders", len(snc_folders))
    col2.metric("📄 Documents", len(snc_docs))
    col3.metric("📦 Boxes", snc_folders["box"].nunique())
    col4.metric("📊 Avg Docs/Folder", f"{snc_folders['doc_count'].mean():.1f}" if len(snc_folders) > 0 else "0")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📁 Folders", "📊 Keywords & Themes", "📅 Timeline", "📈 Statistics"])

    with tab1:
        # Folder list with doc counts
        folder_display = snc_folders[["folder_id", "box", "label", "date", "endDate", "doc_count", "has_scope"]].copy()
        folder_display = folder_display.sort_values("doc_count", ascending=False)
        folder_display["has_scope"] = folder_display["has_scope"].map({True: "✅", False: "❌"})
        st.dataframe(
            folder_display.rename(columns={
                "folder_id": "Folder ID", "box": "Box", "label": "Label",
                "date": "Start", "endDate": "End", "doc_count": "Docs", "has_scope": "Scope",
            }),
            use_container_width=True, hide_index=True, height=400,
        )

        # Expand a folder to see its documents
        st.markdown("### Browse Folder Documents")
        folder_ids = snc_folders["folder_id"].tolist()
        folder_labels = snc_folders.set_index("folder_id")["label"].to_dict()
        selected_folder = st.selectbox(
            "Select Folder",
            folder_ids,
            format_func=lambda x: f"{x} — {folder_labels.get(x, '')}",
        )

        if selected_folder:
            folder_items = snc_docs[snc_docs["folder_id"] == selected_folder]
            st.markdown(f"**{len(folder_items)} documents in this folder:**")
            for _, doc in folder_items.iterrows():
                with st.expander(f"📄 {doc['doc_id']} — {doc['title']}"):
                    if doc["has_summary"]:
                        st.markdown(f"**Summary:** {doc['summary']}")
                    st.caption(f"Date: {doc['date']} | OCR Pages: {doc['ocr_pages']}")

    with tab2:
        st.markdown("### Top Keywords from Document Titles")
        title_texts = snc_docs["title"].dropna().tolist()
        title_kw = extract_keywords(title_texts, top_n=25)
        if title_kw:
            kw_df = pd.DataFrame(title_kw, columns=["Word", "Count"])
            fig = px.bar(
                kw_df, x="Count", y="Word", orientation="h",
                color="Count", color_continuous_scale="Viridis",
            )
            fig = apply_dark_theme(fig, height=500)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough title data for keyword analysis.")

        st.markdown("### Top Keywords from Document Summaries")
        summary_texts = snc_docs["summary"].dropna().tolist()
        summary_kw = extract_keywords(summary_texts, top_n=25)
        if summary_kw:
            kw_df = pd.DataFrame(summary_kw, columns=["Word", "Count"])
            fig = px.bar(
                kw_df, x="Count", y="Word", orientation="h",
                color="Count", color_continuous_scale="Plasma",
            )
            fig = apply_dark_theme(fig, height=500)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough summary data for keyword analysis.")

    with tab3:
        # Timeline of documents
        doc_dates = snc_docs[snc_docs["date"].notna() & (snc_docs["date"] != "")].copy()
        if len(doc_dates) > 0:
            doc_dates["parsed_date"] = pd.to_datetime(doc_dates["date"], errors="coerce")
            doc_dates = doc_dates[doc_dates["parsed_date"].notna()]
            if len(doc_dates) > 0:
                doc_dates["year"] = doc_dates["parsed_date"].dt.year
                year_counts = doc_dates.groupby("year").size().reset_index(name="count")
                fig = px.bar(
                    year_counts, x="year", y="count",
                    color_discrete_sequence=[COLORS["primary"]],
                    labels={"year": "Year", "count": "Documents"},
                    title=f"Documents per Year — {selected_snc}",
                )
                fig = apply_dark_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Monthly heatmap for years with most docs
                doc_dates["month"] = doc_dates["parsed_date"].dt.month
                monthly = doc_dates.groupby(["year", "month"]).size().reset_index(name="count")
                if len(monthly) > 5:
                    fig = px.density_heatmap(
                        monthly, x="month", y="year", z="count",
                        color_continuous_scale="Viridis",
                        labels={"month": "Month", "year": "Year", "count": "Documents"},
                        title="Monthly Document Density",
                    )
                    fig = apply_dark_theme(fig, height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid dates found for timeline.")
        else:
            st.info("No date data available for this SNC.")

    with tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Document Statistics")
            stats = {
                "Total Documents": len(snc_docs),
                "Documents with Titles": snc_docs["has_title"].sum(),
                "Documents with Summaries": snc_docs["has_summary"].sum(),
                "Average OCR Pages": f"{snc_docs['ocr_pages'].mean():.1f}",
                "Max OCR Pages": snc_docs["ocr_pages"].max(),
                "Total OCR Characters": f"{snc_docs['ocr_length'].sum():,}",
            }
            for k, v in stats.items():
                st.markdown(f"- **{k}:** {v}")

        with col_b:
            st.markdown("### Folder Statistics")
            stats_f = {
                "Total Folders": len(snc_folders),
                "Folders with Documents": (snc_folders["doc_count"] > 0).sum(),
                "Empty Folders": (snc_folders["doc_count"] == 0).sum(),
                "Max Docs in a Folder": snc_folders["doc_count"].max(),
                "Min Docs in a Folder": snc_folders["doc_count"].min(),
                "Folders with Scope Note": snc_folders["has_scope"].sum(),
            }
            for k, v in stats_f.items():
                st.markdown(f"- **{k}:** {v}")

        # Doc count per folder bar chart
        fig = px.bar(
            snc_folders.sort_values("doc_count", ascending=False),
            x="folder_id", y="doc_count",
            color="doc_count", color_continuous_scale="Viridis",
            labels={"folder_id": "Folder ID", "doc_count": "Documents"},
            title="Documents per Folder",
            hover_data=["label", "box"],
        )
        fig = apply_dark_theme(fig, height=400)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8rem;">'
    "SUSHI Phase 0 Data Analysis Dashboard · Built with Streamlit & Plotly · 2026"
    "</p>",
    unsafe_allow_html=True,
)
