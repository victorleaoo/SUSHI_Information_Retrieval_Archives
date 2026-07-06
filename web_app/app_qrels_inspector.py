import streamlit as st
import json
import os

# Configure page
st.set_page_config(
    page_title="SUSHI QRELs Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling for robust aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .topic-container {
        background: linear-gradient(145deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .topic-title {
        color: #58a6ff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .topic-section {
        margin-bottom: 16px;
    }
    .topic-section-title {
        color: #8b949e;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metadata-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        height: 100%;
    }
    .metadata-card-title {
        color: #4493f8;
        font-size: 1.2rem;
        font-weight: 600;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    .stJson {
        background-color: #0d1117 !important;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FOLDERS_PATH = os.path.join(BASE_DIR, "data", "folders_metadata", "FoldersV1.3.json")
ITEMS_PATH = os.path.join(BASE_DIR, "data", "items_metadata", "itemsV1.2.json")
ECF_PATH = os.path.join(BASE_DIR, "web_app", "Ntcir18SushiOfficialExperimentControlFileV1.1.json")
QRELS_DOCS_PATH = os.path.join(BASE_DIR, "qrels", "formal-document-qrel.txt")
QRELS_FOLDERS_PATH = os.path.join(BASE_DIR, "qrels", "formal-folder-qrel.txt")

@st.cache_data
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_qrels(path):
    qrels = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    topic_id = parts[0]
                    raw_id = parts[2]
                    # Remove .pdf extension if present
                    item_id = raw_id[:-4] if raw_id.lower().endswith('.pdf') else raw_id
                    try:
                        relevance = int(parts[3])
                    except ValueError:
                        continue
                    if relevance > 0:
                        if topic_id not in qrels:
                            qrels[topic_id] = []
                        qrels[topic_id].append((item_id, relevance))
    except FileNotFoundError:
        pass
    
    # Sort by relevance score descending
    for topic in qrels:
        qrels[topic].sort(key=lambda x: x[1], reverse=True)
        
    return qrels

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🔍 QRELs Inspector")
    st.sidebar.markdown("Study relevant documents and their main folders to guide LLM query expansion.")
    
    with st.spinner("Loading data..."):
        folders_data = load_json(FOLDERS_PATH)
        items_data = load_json(ITEMS_PATH)
        ecf_data = load_json(ECF_PATH)
        qrels_docs = load_qrels(QRELS_DOCS_PATH)
        qrels_folders = load_qrels(QRELS_FOLDERS_PATH)

    # Extract topics
    topics = {}
    if "ExperimentSets" in ecf_data:
        for es in ecf_data["ExperimentSets"]:
            if "Topics" in es:
                topics.update(es["Topics"])
                
    if not topics:
        st.error("No topics found in the ECF file.")
        return

    # Sidebar selection
    topic_ids = sorted(list(topics.keys()))
    selected_topic_id = st.sidebar.selectbox("Select Topic", topic_ids)
    
    topic_data = topics[selected_topic_id]
    
    # Topic Details UI
    st.markdown(f'''
    <div class="topic-container">
        <div class="topic-title">{selected_topic_id}: {topic_data.get("TITLE", "")}</div>
        <div class="topic-section">
            <div class="topic-section-title">Description</div>
            <div>{topic_data.get("DESCRIPTION", "")}</div>
        </div>
        <div class="topic-section">
            <div class="topic-section-title">Narrative</div>
            <div>{topic_data.get("NARRATIVE", "")}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Tabs for Documents and Folders
    tab1, tab2 = st.tabs(["📄 Relevant Documents", "📂 Relevant Folders"])
    
    with tab1:
        docs_for_topic = qrels_docs.get(selected_topic_id, [])
        if not docs_for_topic:
            st.info("No relevant documents found for this topic.")
        else:
            st.markdown(f"**Found {len(docs_for_topic)} relevant documents** (Sorted by relevance)")
            
            # Prepare options for selectbox
            doc_options = {}
            for doc_id, rel in docs_for_topic:
                item_meta = items_data.get(doc_id, {})
                title = item_meta.get("Brown Title") or item_meta.get("NARA Title") or doc_id
                # Truncate title for display
                display_title = str(title)[:80] + "..." if len(str(title)) > 80 else str(title)
                doc_options[f"[{rel}⭐] {doc_id} - {display_title}"] = doc_id
                
            selected_doc_label = st.selectbox("Select a relevant document to inspect:", list(doc_options.keys()))
            selected_doc_id = doc_options[selected_doc_label]
            
            st.divider()
            
            # Split view for Document vs Main Folder
            col1, col2 = st.columns(2)
            
            doc_meta = items_data.get(selected_doc_id, {})
            folder_id = doc_meta.get("Sushi Folder")
            folder_meta = folders_data.get(folder_id, {}) if folder_id else {}
            
            with col1:
                st.markdown('<div class="metadata-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metadata-card-title">📄 Document Fields ({selected_doc_id})</div>', unsafe_allow_html=True)
                
                if doc_meta:
                    st.json(doc_meta, expanded=True)
                else:
                    st.warning("No metadata found for this document.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metadata-card">', unsafe_allow_html=True)
                if folder_id:
                    st.markdown(f'<div class="metadata-card-title">📂 Main Folder Fields ({folder_id})</div>', unsafe_allow_html=True)
                    if folder_meta:
                        st.json(folder_meta, expanded=True)
                    else:
                        st.warning("No metadata found for this folder.")
                else:
                    st.markdown(f'<div class="metadata-card-title">📂 Main Folder Fields</div>', unsafe_allow_html=True)
                    st.warning("This document does not have a linked 'Sushi Folder' in its metadata.")
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        folders_for_topic = qrels_folders.get(selected_topic_id, [])
        if not folders_for_topic:
            st.info("No relevant folders found for this topic.")
        else:
            st.markdown(f"**Found {len(folders_for_topic)} relevant folders** (Sorted by relevance)")
            
            folder_options = {}
            for folder_id, rel in folders_for_topic:
                f_meta = folders_data.get(folder_id, {})
                title = f_meta.get("Brown Title") or f_meta.get("NARA Title") or folder_id
                display_title = str(title)[:80] + "..." if len(str(title)) > 80 else str(title)
                folder_options[f"[{rel}⭐] {folder_id} - {display_title}"] = folder_id
                
            selected_folder_label = st.selectbox("Select a relevant folder to inspect:", list(folder_options.keys()))
            selected_folder_id = folder_options[selected_folder_label]
            
            st.divider()
            
            f_meta = folders_data.get(selected_folder_id, {})
            
            st.markdown('<div class="metadata-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metadata-card-title">📂 Folder Fields ({selected_folder_id})</div>', unsafe_allow_html=True)
            if f_meta:
                st.json(f_meta, expanded=True)
            else:
                st.warning("No metadata found for this folder.")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
