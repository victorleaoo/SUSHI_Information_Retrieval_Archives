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
    # ==========================================
    # PART 1: VISUAL ANALYSIS (Model Comparison)
    # ==========================================
    st.title("🔬 Experiment Analysis: Model Comparison")
    
    # 1. Scan and Group Runs for the Dropdown
    grouped_runs = u1.get_grouped_run_configurations()
    
    if not grouped_runs:
        st.error("No valid run folders found in ../all_runs/.")
        return

    config_options = sorted(list(grouped_runs.keys()))
    
    st.caption("Select a configuration to compare how different models performed under that specific setting.")
    
    # SINGLE SELECTION for the Chart
    selected_config = st.selectbox(
        "Select Configuration for Visualization:",
        options=config_options,
        index=0 if config_options else None,
        help="Loads the Dumbbell Chart and Stats for all models within this specific configuration."
    )

    if selected_config:
        # Process data ONLY for this config for the visualizations
        df_chart, df_table_dummy, all_topics, model_results = u1.process_experiment_data([selected_config], grouped_runs)

        # --- DYNAMIC SCORE CARDS ---
        st.subheader("Global Performance (Mean nDCG@5)")
        sorted_models = sorted(model_results.keys(), key=lambda x: model_results[x]['stats']['val'], reverse=True)
    
        cols = st.columns(min(len(sorted_models), 4))
        for i, model_key in enumerate(sorted_models):
            stats = model_results[model_key]['stats']
            count = model_results[model_key]['count']
            
            with cols[i % 4]:
                st.metric(
                    # UPDATE: Added (N={count}) to the label for visibility
                    label=f"{model_key} (N={count})", 
                    value=f"{stats['val']:.4f}",
                    delta=f"± {stats['margin']:.4f}",
                    delta_color="off",
                    help=f"Mean nDCG@5 aggregated from {count} random seed executions."
                )

        # --- DUMBELL CHART ---
        if all_topics:
            with st.expander("📈 Model Comparison Chart (nDCG@5)", expanded=True):
                render_charts(df_chart, all_topics)
        else:
            st.warning("No topic data found for this configuration.")

    # ==========================================
    # PART 2: DETAILED COMPARISON TABLE (Global)
    # Focus: Compare ANY run against ANY run
    # ==========================================
    st.markdown("---")
    st.header("📑 Detailed Comparison Table (Cross-Experiment)")
    st.caption("Select specific run folders to compare their detailed metrics side-by-side.")

    # 1. Get ALL Run Folders (Flat List)
    all_runs_df = u1.get_all_runs_statistics() 
    if all_runs_df.empty:
        st.warning("No runs found.")
        return

    all_run_names = all_runs_df['Run Name'].tolist()

    # 2. Smart Defaults: Pre-select runs from the config chosen above (if any)
    # This connects the two parts nicely without restricting the user.
    default_selection = []
    if selected_config and selected_config in grouped_runs:
        default_selection = grouped_runs[selected_config]
    
    # Fallback if list is empty or too long
    if not default_selection: 
        default_selection = all_run_names[:3]

    # 3. GLOBAL MULTI-SELECT
    sel_runs_table = st.multiselect(
        "Select Runs to Compare:",
        options=all_run_names,
        default=default_selection,
        help="You can add runs from different configurations here to compare them."
    )

    if sel_runs_table:
        # 4. Generate the Unified Dataframe for specific folders
        df_unified = u1.get_unified_comparison_dataframe(sel_runs_table)
        
        # 5. Filter Topics Columns (Reuse the topics found in the chart logic if available)
        # Or just show all columns available in the dataframe
        fixed_cols = ["Experiment Folder", "Global nDCG@5", "Global Relevance"]
        
        # Get topic columns that exist in the dataframe
        # We sort them naturally (T1, T2, T10...)
        available_cols = [c for c in df_unified.columns if c not in fixed_cols]
        # Sort using the natural keys helper from u1
        available_cols.sort(key=u1.natural_keys)
        
        final_cols = fixed_cols + available_cols
        
        df_display = df_unified[final_cols]

        # 6. Render
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Experiment Folder": st.column_config.TextColumn("Experiment Folder", width="medium"),
                "Global nDCG@5": st.column_config.TextColumn("Global nDCG", width="small"),
                "Global Relevance": st.column_config.TextColumn("Global Rel.", width="small")
            }
        )
    else:
        st.info("Select experiments to view the comparison table.")

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
                                st.markdown(f"**Document Sushi Folder Metadata:**")
                                f_meta = folders_meta.get(sushi_folder_id, {})
                                if f_meta:
                                    st.json(f_meta)
                                else: st.warning("No metadata found.")

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