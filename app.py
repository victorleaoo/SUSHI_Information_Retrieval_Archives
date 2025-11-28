import streamlit as st
import os
import json
import base64

# Configuração da página
st.set_page_config(layout="wide", page_title="Sistema de Visualização")

# --- CONFIGURAÇÃO DE CAMINHOS (AJUSTE AQUI) ---
PATH_FOLDERS_JSON = 'data/folders_metadata/FoldersV1.2.json'
PATH_ITEMS_JSON = 'data/items_metadata/itemsV1.2.json'
PATH_TOPICS_JSON = 'ecf/Ntcir18SushiOfficialExperimentControlFileV1.1.json'

# --- CAMINHOS DOS 3 ARQUIVOS DE RELEVÂNCIA ---
# Ajuste conforme os nomes reais dos seus arquivos
PATH_QRELS_DOCS = 'qrels/formal-run-qrels/formal-document-qrel.txt'
PATH_QRELS_FOLDERS = 'qrels/formal-run-qrels/formal-folder-qrel.txt'
PATH_QRELS_BOXES = 'qrels/formal-run-qrels/formal-box-qrel.txt'

BASE_DIR_FILES = os.path.join("data", "raw")

# --- FUNÇÕES DE CARREGAMENTO (BACKEND) ---

@st.cache_data
def load_metadata():
    """Carrega metadados de pastas e itens."""
    folders_data, items_data = {}, {}
    try:
        with open(PATH_FOLDERS_JSON, 'r', encoding='utf-8') as f:
            folders_data = json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {PATH_FOLDERS_JSON}")
    
    try:
        with open(PATH_ITEMS_JSON, 'r', encoding='utf-8') as f:
            items_data = json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {PATH_ITEMS_JSON}")

    return folders_data, items_data

@st.cache_data
def load_topics_data():
    """Lê o JSON e retorna um dicionário com todos os tópicos encontrados."""
    all_topics = {}
    try:
        with open(PATH_TOPICS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "ExperimentSets" in data:
                for experiment_set in data["ExperimentSets"]:
                    if "Topics" in experiment_set:
                        all_topics.update(experiment_set["Topics"])
            else:
                st.warning("'ExperimentSets' not found in topics file.")
    except FileNotFoundError:
        st.error(f"Topics file not found: {PATH_TOPICS_JSON}")
    except json.JSONDecodeError:
        st.error("Error decoding topics JSON.")
    return all_topics

@st.cache_data
def load_qrels_data(filepath):
    """
    Lê um arquivo de relevância (formato: topic 0 item_id rel).
    Retorna um dict: {topic_id: [(item_id, rel_score), ...]}
    """
    qrels = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    topic_id = parts[0]
                    item_id = parts[2]
                    try:
                        relevance = int(parts[3])
                    except ValueError:
                        continue 
                    
                    # Filtra apenas relevantes (>0)
                    if relevance > 0:
                        if topic_id not in qrels:
                            qrels[topic_id] = []
                        qrels[topic_id].append((item_id, relevance))
    except FileNotFoundError:
        # Retorna vazio se o arquivo não existir, para não quebrar o app
        # (pode ser que você ainda não tenha gerado um dos arquivos)
        return {}
    except Exception as e:
        st.error(f"Error reading qrels file {filepath}: {e}")
    return qrels

def display_pdf(file_path):
    """Gera iframe para PDF."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error opening PDF: {e}")

# --- INTERFACE DO USUÁRIO ---

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the module:", ["📂 Document/Folder Explorer", "🔍 Topic Viewer"])
st.sidebar.divider()

# Carregar metadados globais
folders_meta, items_meta = load_metadata()

# ==========================================
# MÓDULO 1: EXPLORADOR DE ARQUIVOS
# ==========================================
if app_mode == "📂 Document/Folder Explorer":
    st.title("📂 Document/Folder Explorer")
    
    with st.sidebar:
        st.header("File Selection")
        if os.path.exists(BASE_DIR_FILES):
            boxes = [d for d in os.listdir(BASE_DIR_FILES) if os.path.isdir(os.path.join(BASE_DIR_FILES, d))]
            boxes.sort()
            selected_box = st.selectbox("Select the Box:", options=[""] + boxes)
        else:
            st.error(f"Base directory '{BASE_DIR_FILES}' not found.")
            selected_box = None

        selected_folder = None
        if selected_box:
            box_path = os.path.join(BASE_DIR_FILES, selected_box)
            folders = [d for d in os.listdir(box_path) if os.path.isdir(os.path.join(box_path, d))]
            folders.sort()
            selected_folder = st.selectbox("Select the folder:", options=[""] + folders)

        selected_file = None
        if selected_folder:
            folder_path = os.path.join(BASE_DIR_FILES, selected_box, selected_folder)
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            files.sort()
            selected_file = st.selectbox("Select the Document:", options=[""] + files)

    if selected_box and selected_folder:
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.subheader("📋 Information")
            st.markdown("### About the Folder")
            f_meta = folders_meta.get(selected_folder)
            if f_meta:
                st.json(f_meta)
            else:
                st.warning(f"No metadata for: {selected_folder}")

            st.divider()

            if selected_file:
                st.markdown("### About the Document")
                file_key = os.path.splitext(selected_file)[0]
                i_meta = items_meta.get(file_key)
                if i_meta:
                    if "summary" in i_meta:
                        st.info(f"**Summary:** {i_meta['summary']}")
                    with st.expander("View all data (JSON)"):
                        st.json(i_meta)
                    if "ocr" in i_meta and i_meta["ocr"]:
                        with st.expander("View OCR"):
                            st.text(i_meta["ocr"][0])
                else:
                    st.warning(f"No metadata for: {file_key}")
            else:
                st.info("Select a document to see details.")

        with col2:
            if selected_file:
                st.subheader("📄 PDF Visualizer")
                full_file_path = os.path.join(BASE_DIR_FILES, selected_box, selected_folder, selected_file)
                display_pdf(full_file_path)
            else:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.info("👈 Select a PDF file.")
    elif selected_box:
        st.info("👈 Select a folder.")
    else:
        st.info("👈 Use the sidebar to navigate the files.")


# ==========================================
# MÓDULO 2: VISUALIZADOR DE TÓPICOS
# ==========================================
elif app_mode == "🔍 Topic Viewer":
    st.markdown("# 🔍 [Definition of Research Topics](https://victorleaoo.github.io/SUSHI_Information_Retrieval_Archives/3_sushi_solutions/2_topics/)")
    
    topics_dict = load_topics_data()
    
    # Carrega os 3 arquivos separadamente
    qrels_docs = load_qrels_data(PATH_QRELS_DOCS)
    qrels_folders = load_qrels_data(PATH_QRELS_FOLDERS)
    qrels_boxes = load_qrels_data(PATH_QRELS_BOXES)
    
    if topics_dict:
        topic_ids = list(topics_dict.keys())
        topic_ids.sort()
        
        with st.sidebar:
            st.header("Topic Selection")
            selected_topic_id = st.selectbox("Choose the Topic ID:", options=topic_ids)
        
        if selected_topic_id:
            topic_data = topics_dict[selected_topic_id]
            
            with st.container():
                st.subheader(f"📌 {selected_topic_id}")
                st.markdown("#### Title")
                st.info(topic_data.get('TITLE', 'No title.'))
                
                col_desc, col_narr = st.columns(2)
                with col_desc:
                    st.markdown("**Description**")
                    st.caption(topic_data.get('DESCRIPTION', 'No description.'))
                with col_narr:
                    st.markdown("**Narrative**")
                    st.warning(topic_data.get('NARRATIVE', 'No narrative.'))
                
                with st.expander("View Raw JSON of this Topic"):
                    st.json(topic_data)
                
                st.divider()
                st.markdown("### 🎯 Relevant Items")

                # Recupera as listas específicas para este tópico
                rel_docs_list = qrels_docs.get(selected_topic_id, [])
                rel_folders_list = qrels_folders.get(selected_topic_id, [])
                rel_boxes_list = qrels_boxes.get(selected_topic_id, [])

                # Cria as abas com a contagem correta
                tab_docs, tab_folders, tab_boxes = st.tabs([
                    f"📄 Documents ({len(rel_docs_list)})", 
                    f"📁 Folders ({len(rel_folders_list)})", 
                    f"📦 Boxes ({len(rel_boxes_list)})"
                ])

                # --- ABA DOCUMENTOS ---
                with tab_docs:
                    if rel_docs_list:
                        for doc_id, score in rel_docs_list:
                            score_label = "⭐⭐⭐ High (3)" if score == 3 else "⭐ Relevant (1)"
                            
                            # Tenta pegar metadados para mostrar título
                            title_display = doc_id
                            if doc_id in items_meta:
                                meta = items_meta[doc_id]
                                title_display = meta.get('title') or meta.get('Brown Title') or doc_id
                            
                            st.markdown(f"**{doc_id}** - {score_label}")
                            st.caption(f"Title: {title_display}")
                            st.markdown("---")
                    else:
                        st.caption("No relevant documents identified in qrels.")

                # --- ABA PASTAS ---
                with tab_folders:
                    if rel_folders_list:
                        for folder_id, score in rel_folders_list:
                            score_label = "⭐⭐⭐ High (3)" if score == 3 else "⭐ Relevant (1)"
                            
                            # Tenta pegar metadados para mostrar label
                            label_display = folder_id
                            if folder_id in folders_meta:
                                meta = folders_meta[folder_id]
                                label_display = meta.get('label') or folder_id
                                
                            st.markdown(f"**{folder_id}** - {score_label}")
                            st.caption(f"Label: {label_display}")
                            st.markdown("---")
                    else:
                        st.caption("No relevant folders identified in qrels.")

                # --- ABA CAIXAS ---
                with tab_boxes:
                    if rel_boxes_list:
                        for box_id, score in rel_boxes_list:
                            score_label = "⭐⭐⭐ High (3)" if score == 3 else "⭐ Relevant (1)"
                            st.markdown(f"**{box_id}** - {score_label}")
                            # Caixas geralmente não têm metadados extras além do ID/Nome
                            st.markdown("---")
                    else:
                        st.caption("No relevant boxes identified in qrels.")

    else:
        st.info("No topics found or file not loaded.")