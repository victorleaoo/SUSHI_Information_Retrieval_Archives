import streamlit as st
import pandas as pd
import altair as alt

# Importação dos Módulos
import utils_experiments_viz as u1
import utils_topics_viz as u2

st.set_page_config(layout="wide", page_title="SUSHI Research Platform")

# ==========================================
# UI COMPONENTS - APP 1 (Experiment Analyzer)
# ==========================================

def render_charts(df_chart: pd.DataFrame, topics_to_display: list):
    if df_chart.empty or not topics_to_display: return
    
    chart_data = df_chart[df_chart['Topic'].isin(topics_to_display)]
    
    color_scale = alt.Color(
        'Type', 
        scale=alt.Scale(domain=list(u1.COLOR_MAP.keys()), range=list(u1.COLOR_MAP.values())), 
        legend=alt.Legend(title="Metric Type", orient="top")
    )
    
    base = alt.Chart(chart_data).encode(y=alt.Y('Topic', title="Topics", sort=topics_to_display))
    
    rule_bg = base.mark_line(color='lightgray', strokeDash=[2,2], opacity=0.3).encode(
        x='min(nDCG)', x2='max(nDCG)', detail='Topic'
    )
    ci_rule = base.mark_rule(opacity=0.6, thickness=2).encode(
        x='min_ci', x2='max_ci', color=color_scale
    )
    tick_min = base.mark_tick(thickness=2, height=12).encode(x='min_ci', color=color_scale)
    tick_max = base.mark_tick(thickness=2, height=12).encode(x='max_ci', color=color_scale)
    points = base.mark_circle(size=120, opacity=1).encode(
        x=alt.X('nDCG', title="Mean nDCG@5 with 95% CI"), 
        color=color_scale, 
        tooltip=['Topic', 'Type', 'nDCG', 'min_ci', 'max_ci']
    )
    
    final_chart = (rule_bg + ci_rule + tick_min + tick_max + points).properties(height=len(topics_to_display) * 65)
    st.altair_chart(final_chart)

def render_table(df_table: pd.DataFrame, topics_to_display: list):
    if df_table.empty: return
    
    # 1. Filter Columns
    cols_to_keep = ["Experiment"] + topics_to_display
    cols_to_keep = [c for c in cols_to_keep if c in df_table.columns]
    df_filtered = df_table[cols_to_keep]
    
    # 2. Get CSS (Now returns empty string from Utils)
    css = u1.generate_column_css(df_filtered)
    
    # 3. Define Clean Styles (Removed Green Header)
    # Added simple borders and alternating rows for readability without strong colors
    st.markdown(f"""
    <style>
        {css} 
        .dataframe {{
            width: 100%;
            border-collapse: collapse;
        }}
        .dataframe th {{
            background-color: #f0f2f6; /* Neutral Streamlit Gray */
            color: black;
            font-weight: bold;
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }} 
        .dataframe td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        /* Highlight the 'Avg' rows slightly for differentiation */
        tr:contains('>>') {{
            background-color: #f9f9f9 !important; 
            font-weight: bold; 
            border-top: 2px solid #ccc;
        }}
    </style>""", unsafe_allow_html=True)
    
    # 4. Wrap in Expander
    # expanded=False makes it collapsed by default (saving screen space)
    with st.expander("📝 Detailed Results Table", expanded=False):
        st.write(df_filtered.to_html(escape=False, index=False, classes="dataframe"), unsafe_allow_html=True)

def run_experiment_analyzer_ui():
    st.sidebar.header("Experiment Filters")
    # 1. SCAN FOLDERS
    options, _ = u1.scan_available_runs()
    
    if not options["search"] or not options["model"]:
        st.error("No valid run folders found in ../all_runs/. Check naming convention: <Search>_<Exp>_<Query>_<Model>")
        return

    # 2. DYNAMIC FILTERS
    # Use index=0 safely if list not empty
    sel_search = st.sidebar.selectbox("Searching Field", options["search"], index=0)
    sel_query = st.sidebar.selectbox("Query Field", options["query"], index=0)
    sel_model = st.sidebar.selectbox("Model", options["model"], index=0)
    
    # Expansion Strategy (The 'Blue' line)
    # We default to the first available expansion strategy found
    if not options["expansion"]:
        st.sidebar.warning("No Expansion (EX) runs found. Comparisons may be empty.")
        sel_expansion = "None"
    else:
        sel_expansion = st.sidebar.selectbox("Expansion Strategy", options["expansion"], index=0)

    # 3. PROCESS DATA (Pass all 4 parameters)
    df_chart, df_table, all_topics, stats_ex, stats_nex, stats_oracle, stats_emb = u1.process_experiment_data(
        sel_search, sel_query, sel_model, sel_expansion
    )

    sel_topic_filter = st.sidebar.radio("Filter Topics:", ["All Topics", "Difficult Topics Only", "Impossible Topics Only"])

    run_type_options = {
        "With Expansion": "Avg With Expansion",
        "No Expansion": "Avg No Expansion",
        "All Training Docs": "AllTrainingDocs",
        "SUSHISubmissions": "SUSHISubmissions"
    }
    if stats_emb:
        run_type_options["Embeddings"] = "Embeddings (F_EMB_T)"

    sel_run_types = st.sidebar.pills("Show Runs:", options=list(run_type_options.keys()), default=list(run_type_options.keys()), selection_mode="multi")
    selected_internal_types = [run_type_options[k] for k in sel_run_types]

    sort_opt = st.sidebar.radio("Sort Topics By (Desc nDCG):", ["Default (Topic ID)", "Sort by TrainingDocuments", "Sort by SUSHISubmissions", "Sort by Mean With Exp"])

    st.title("🔬 Analysis of Experiments")
    st.caption(f"Comparing: `{sel_search}_{sel_expansion}_{sel_query}_{sel_model}` (Blue) vs `{sel_search}_NEX_{sel_query}_{sel_model}` (Orange)")

    # --- Score Cards ---
    if stats_emb: c1, c2, c3, c4 = st.columns([1,1,1,1])
    else: 
        c1, c2, c3 = st.columns([1,1,1])
        c4 = None

    with c1:
        st.markdown("**Mean nDCG@5 (With Exp):**")
        st.info(f"{stats_ex['val']:.4f} ± {stats_ex['margin']:.4f}", icon="🔵")
    with c2:
        st.markdown("**Mean nDCG@5 (No Exp):**")
        st.warning(f"{stats_nex['val']:.4f} ± {stats_nex['margin']:.4f}", icon="🟠")
    with c3:
        st.markdown("**Mean nDCG@5 (All Training):**")
        st.success(f"{stats_oracle['val']:.4f} ± {stats_oracle['margin']:.4f}", icon="🟢")
    if c4 and stats_emb:
        with c4:
            st.markdown("**Mean nDCG@5 (Embeddings):**")
            st.error(f"{stats_emb['val']:.4f} ± {stats_emb['margin']:.4f}", icon="🟣")

    if df_chart.empty:
        st.error("No data found for this configuration.")
        return

    # --- Filter ---
    filtered_topics = []
    if sel_topic_filter == "All Topics":
        filtered_topics = all_topics
    elif sel_topic_filter == "Difficult Topics Only":
        filtered_topics = [t for t in all_topics if u1.get_topic_number(t) in u1.DIFFICULT_TOPICS]
    elif sel_topic_filter == "Impossible Topics Only":
        filtered_topics = [t for t in all_topics if u1.get_topic_number(t) in u1.IMPOSSIBLE_TOPICS]

    df_chart = df_chart[df_chart['Type'].isin(selected_internal_types)]

    def should_keep_table_row(row_name: str) -> bool:
        if "With Exp" in row_name or "(EX)" in row_name: return "With Expansion" in sel_run_types
        if "Without Exp" in row_name or "(NEX)" in row_name or "No Exp" in row_name: return "No Expansion" in sel_run_types
        if "All Training" in row_name: return "All Training Docs" in sel_run_types
        if "SUSHISubmissions" in row_name: return "SUSHISubmissions" in sel_run_types
        return True 
    df_table = df_table[df_table['Experiment'].apply(should_keep_table_row)]

    # Sorting
    def get_sort_value(topic: str, run_type: str) -> float:
        row = df_chart[(df_chart['Topic'] == topic) & (df_chart['Type'] == run_type)]
        if not row.empty: return row.iloc[0]['nDCG']
        return -1.0

    topics_to_display = filtered_topics
    if sort_opt == "Default (Topic ID)":
        topics_to_display = sorted(filtered_topics, key=u1.natural_keys)
    elif sort_opt == "Sort by TrainingDocuments":
        topics_to_display = sorted(filtered_topics, key=lambda t: get_sort_value(t, "AllTrainingDocs"), reverse=True)
    elif sort_opt == "Sort by SUSHISubmissions":
        topics_to_display = sorted(filtered_topics, key=lambda t: get_sort_value(t, "SUSHISubmissions"), reverse=True)
    elif sort_opt == "Sort by Mean With Exp":
        topics_to_display = sorted(filtered_topics, key=lambda t: get_sort_value(t, "Avg With Expansion"), reverse=True)

    if topics_to_display:
        with st.expander("📈 Expansion Impact Analysis", expanded=True):
            render_charts(df_chart, topics_to_display)
        st.divider()
        render_table(df_table, topics_to_display)
    else:
        st.info(f"No topics found for the selected filter: {sel_topic_filter}")

    st.markdown("---")
    st.header("📊 Runs Comparisons")
    
    df_runs = u1.get_all_runs_statistics()
    
    if not df_runs.empty:
        all_run_names = df_runs['Run Name'].tolist()
        
        # Smart Filter: Default to showing runs that match current selection (EX and NEX)
        current_context_runs = [
            f"{sel_search}_{sel_expansion}_{sel_query}_{sel_model}",
            f"{sel_search}_NEX_{sel_query}_{sel_model}"
        ]
        # Filter to only existing ones
        default_selection = [r for r in current_context_runs if r in all_run_names]
        # If none match, show top 5
        if not default_selection:
            default_selection = all_run_names[:5]

        sel_runs_compare = st.multiselect(
            "Select Runs to Compare:", 
            options=all_run_names, 
            default=default_selection
        )
        
        # 3. Filter DataFrame based on selection
        df_runs_filtered = df_runs[df_runs['Run Name'].isin(sel_runs_compare)]
        
        # 4. Format the Metrics Column (Mean ± Margin)
        # We create a new column for display purposes
        df_runs_filtered['nDCG@5 (Mean ± Margin)'] = df_runs_filtered.apply(
            lambda x: f"{x['Mean']:.4f} ± {x['Margin']:.4f}", axis=1
        )
        
        # 5. Display
        # Reorder columns for cleaner view
        display_cols = ['Run Name', 'nDCG@5 (Mean ± Margin)', 'Random Runs Count']
        
        st.dataframe(
            df_runs_filtered[display_cols], 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Run Name": st.column_config.TextColumn("Experiment Folder", width="medium"),
                "nDCG@5 (Mean ± Margin)": st.column_config.TextColumn("Global Performance", width="small"),
                "Random Runs Count": st.column_config.NumberColumn("# Random ECFs", help="Number of random runs found in folder")
            }
        )
    else:
        st.warning("No run statistics found in ../all_runs/")

# ==========================================
# UI COMPONENTS - APP 2 (SUSHI Visualization)
# ==========================================

def run_sushi_visualization_ui():
    st.sidebar.title("SUSHI Visual Controls")
    
    folders_meta, items_meta = u2.load_metadata()
    ecf_data = u2.load_ecf_data()
    q_docs = u2.load_qrels_data(u2.PATH_QRELS_DOCS)
    q_folders = u2.load_qrels_data(u2.PATH_QRELS_FOLDERS)
    q_boxes = u2.load_qrels_data(u2.PATH_QRELS_BOXES)

    st.title("🔍 Topic Viewer & Content Explorer")

    all_topics = {}
    if "ExperimentSets" in ecf_data:
        for es in ecf_data["ExperimentSets"]:
            if "Topics" in es: all_topics.update(es["Topics"])
            
    topic_ids = sorted(list(all_topics.keys()))
    sel_topic = st.sidebar.selectbox("Select Topic ID:", topic_ids)

    if sel_topic:
        t_data = all_topics[sel_topic]
        with st.expander("Topic Details", expanded=True):
            st.header(f"{sel_topic}: {t_data.get('TITLE','')}")
            c1, c2 = st.columns([1, 2])
            c1.info(f"**Description:**\n{t_data.get('DESCRIPTION','')}")
            c2.warning(f"**Narrative:**\n{t_data.get('NARRATIVE','')}")
        
        st.divider()
        rd = q_docs.get(sel_topic, [])
        rf = q_folders.get(sel_topic, [])
        rb = q_boxes.get(sel_topic, [])
        
        tab_docs, tab_folders, tab_boxes = st.tabs([f"📄 Documents ({len(rd)})", f"📂 Folders ({len(rf)})", f"📦 Boxes ({len(rb)})"])
        
        with tab_docs:
            if not rd:
                st.write("No relevant documents found.")
            else:
                col_list, col_detail = st.columns([1, 2])
                with col_list:
                    st.subheader("Select Document")
                    doc_options = {f"{did} ({'⭐'*sc}) - {u2.get_smart_title(items_meta.get(did), did)[:40]}...": did for did, sc in rd}
                    selected_option = st.selectbox("Relevant Documents:", options=list(doc_options.keys()))
                    selected_doc_id = doc_options[selected_option]
                    score = next((sc for did, sc in rd if did == selected_doc_id), 0)
                    st.caption(f"Relevance Score: {score}")

                with col_detail:
                    st.subheader("Document Content & Metadata")
                    if selected_doc_id:
                        meta = items_meta.get(selected_doc_id, {})
                        sub_t1, sub_t2 = st.tabs(["👁️ PDF Preview", "ℹ️ Metadata"])
                        
                        path = u2.get_file_path_from_metadata(selected_doc_id, items_meta, folders_meta)
                        
                        with sub_t1:
                            b64_pdf = u2.get_pdf_base64(path) if path else None
                            if b64_pdf:
                                pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                                st.markdown(pdf_display, unsafe_allow_html=True)
                            else:
                                st.warning("PDF not found or could not be loaded.")
                        
                        with sub_t2:
                            st.markdown(f"**Title:** {u2.get_smart_title(meta, selected_doc_id)}")
                            if "summary" in meta: st.info(f"**Summary:** {meta['summary']}")
                            st.json(meta)
                            if "ocr" in meta and meta["ocr"]:
                                with st.expander("Show OCR Text"): st.text(meta["ocr"][0])

                            sushi_folder_id = meta.get("Sushi Folder")
                            if sushi_folder_id:
                                specific_folder_meta = folders_meta.get(sushi_folder_id, {})
                                folder_label = specific_folder_meta.get("folder_label", "N/A")
                                st.markdown(f"**Sushi Folder Label:** {folder_label}")

        with tab_folders:
            if not rf: st.write("No relevant folders found.")
            else:
                col_f_list, col_f_detail = st.columns([1, 2])
                with col_f_list:
                    st.subheader("Select Folder")
                    folder_options = {f"{fid} ({'⭐'*sc})": fid for fid, sc in rf}
                    sel_folder_opt = st.selectbox("Relevant Folders:", options=list(folder_options.keys()))
                    selected_folder_id = folder_options[sel_folder_opt]
                with col_f_detail:
                    st.subheader("Folder Metadata")
                    if selected_folder_id:
                        f_meta = folders_meta.get(selected_folder_id, {})
                        if f_meta:
                            if 'label' in f_meta: st.markdown(f"**Label:** {f_meta['label']}")
                            if 'Box' in f_meta: st.markdown(f"**Box:** {f_meta['Box']}")
                            st.json(f_meta)
                        else: st.warning("No metadata found.")

        with tab_boxes:
            for bid, sc in rb:
                st.markdown(f"- **{bid}** ({'⭐'*sc})")

# ==========================================
# MAIN NAVIGATION
# ==========================================

def main():
    st.sidebar.title("📱 App Navigation")
    app_mode = st.sidebar.radio(
        "Choose Application:",
        ["Experiment Analyzer", "Topics and Data Visualizer"]
    )
    
    st.sidebar.markdown("---")
    
    if app_mode == "Experiment Analyzer":
        run_experiment_analyzer_ui()
    elif app_mode == "Topics and Data Visualizer":
        run_sushi_visualization_ui()

if __name__ == "__main__":
    main()