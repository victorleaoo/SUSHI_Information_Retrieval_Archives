import streamlit as st
import os
import json
import base64

# Configuração da página
st.set_page_config(layout="wide", page_title="SUSHI visualization")

BASE_DIR_FILES = os.path.join("data", "raw")

# Metadados
PATH_FOLDERS_JSON = 'data/folders_metadata/FoldersV1.2.json'
PATH_ITEMS_JSON = 'data/items_metadata/itemsV1.2.json'

# Arquivos de Tópicos e Experimentos
PATH_TOPICS_JSON = 'ecf/formal_run/Ntcir18SushiOfficialExperimentControlFileV1.1.json'
PATH_RUN_RESULTS = 'src/lastest_runs/results/topics/TopicsFolderRun.json'
PATH_TOPICS_METRICS = 'src/lastest_runs/results/topics/TopicsFolderMetrics.json'

# Arquivos de Relevância (Qrels)
PATH_QRELS_DOCS = 'qrels/formal-run-qrels/formal-document-qrel.txt'
PATH_QRELS_FOLDERS = 'qrels/formal-run-qrels/formal-folder-qrel.txt'
PATH_QRELS_BOXES = 'qrels/formal-run-qrels/formal-box-qrel.txt'


# ==========================================
# FUNÇÕES DE CARREGAMENTO (BACKEND)
# ==========================================

@st.cache_data
def load_metadata():
    """Carrega metadados de pastas e itens."""
    folders_data, items_data = {}, {}
    try:
        with open(PATH_FOLDERS_JSON, 'r', encoding='utf-8') as f:
            folders_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Metadata file not found: {PATH_FOLDERS_JSON}")
    
    try:
        with open(PATH_ITEMS_JSON, 'r', encoding='utf-8') as f:
            items_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Metadata file not found: {PATH_ITEMS_JSON}")

    return folders_data, items_data

@st.cache_data
def load_ecf_data():
    """Carrega o arquivo de controle de experimentos (ECF)."""
    try:
        with open(PATH_TOPICS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"ECF file not found: {PATH_TOPICS_JSON}")
        return {}

@st.cache_data
def load_run_results():
    """Carrega o arquivo de resultados (TopicsFolderRun)."""
    try:
        with open(PATH_RUN_RESULTS, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_topics_metrics():
    """Carrega o arquivo de métricas por tópico."""
    try:
        with open(PATH_TOPICS_METRICS, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content: return {}
            try: return json.loads(content)
            except json.JSONDecodeError: return {}
    except FileNotFoundError:
        return {}

@st.cache_data
def load_qrels_data(filepath):
    """
    Lê arquivo de relevância formato TREC.
    Retorna: {topic_id: [(item_id, score), ...]}
    """
    qrels = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    topic_id = parts[0]
                    # ID do item (Doc ou Folder)
                    # Se for doc (tem extensão .pdf), removemos. Se for folder, mantemos.
                    raw_id = parts[2]
                    item_id = raw_id[:-4] if raw_id.lower().endswith('.pdf') else raw_id
                    
                    try: relevance = int(parts[3])
                    except ValueError: continue
                    
                    # Filtra apenas relevantes (>0)
                    if relevance > 0:
                        if topic_id not in qrels: qrels[topic_id] = []
                        qrels[topic_id].append((item_id, relevance))
    except FileNotFoundError:
        return {}
    return qrels

def get_smart_title(item_data, item_id):
    if not item_data: return item_id
    brown_title = item_data.get("Brown Title")
    nara_title = item_data.get("NARA Title")
    if brown_title is not None and not isinstance(brown_title, float): return brown_title
    if nara_title is not None and not isinstance(nara_title, float):
        start = nara_title.find('Concerning')
        if start != -1: nara_title = nara_title[start+11:]
        end1 = nara_title.rfind(':')
        end2 = nara_title.rfind('(')
        end = min(end1, end2)
        if end != -1: nara_title = nara_title[:end]
        return nara_title
    return item_id

def display_pdf(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error opening PDF: {e}")
    else:
        st.warning(f"File not found on disk: {file_path}")

def parse_training_docs(training_docs_list):
    """
    Processa a lista de paths do TrainingDocuments.
    Retorna:
    1. valid_doc_ids: Set com IDs dos Docs
    2. valid_folder_ids: Set com IDs das Pastas (NOVO)
    3. docs_by_folder: Map para navegação rápida
    4. doc_path_map: Map ID -> Path relativo
    """
    doc_ids_set = set()
    folder_ids_set = set() # Novo set para validar pastas
    folder_map = {}
    doc_path_map = {} 

    for path in training_docs_list:
        parts = path.split('/')
        if len(parts) >= 3:
            box = parts[0]
            folder = parts[1]
            filename = parts[-1]
            
            doc_id = os.path.splitext(filename)[0]
            doc_ids_set.add(doc_id)
            folder_ids_set.add(folder)
            
            if folder not in folder_map: folder_map[folder] = []
            folder_map[folder].append({'doc_id': doc_id, 'path': path})
            
            doc_path_map[doc_id] = path
            
    return doc_ids_set, folder_ids_set, folder_map, doc_path_map

def resolve_folder_id(doc_id, items_meta):
    meta = items_meta.get(doc_id, {})
    return meta.get('Sushi Folder')

# ==========================================
# INTERFACE DO USUÁRIO
# ==========================================

st.sidebar.title("SUSHI Visualization")
app_mode = st.sidebar.radio("Module:", [
    "📂 Document/Folder Explorer", 
    "🔍 Topic Viewer", 
    "📊 Experiment Results"
])
st.sidebar.divider()

folders_meta, items_meta = load_metadata()

# ==========================================
# MÓDULO 1: EXPLORADOR DE ARQUIVOS
# ==========================================
if app_mode == "📂 Document/Folder Explorer":
    st.title("📂 Document/Folder Explorer")
    # ... (Código igual ao anterior) ...
    with st.sidebar:
        st.header("File Navigation")
        if os.path.exists(BASE_DIR_FILES):
            boxes = [d for d in os.listdir(BASE_DIR_FILES) if os.path.isdir(os.path.join(BASE_DIR_FILES, d))]
            boxes.sort()
            selected_box = st.selectbox("Box:", options=[""] + boxes)
        else:
            st.error(f"Base dir {BASE_DIR_FILES} not found.")
            selected_box = None
        selected_folder = None
        if selected_box:
            box_path = os.path.join(BASE_DIR_FILES, selected_box)
            folders = [d for d in os.listdir(box_path) if os.path.isdir(os.path.join(box_path, d))]
            folders.sort()
            selected_folder = st.selectbox("Folder:", options=[""] + folders)
        selected_file = None
        if selected_folder:
            folder_path = os.path.join(BASE_DIR_FILES, selected_box, selected_folder)
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            files.sort()
            selected_file = st.selectbox("Document:", options=[""] + files)

    if selected_box and selected_folder:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("Folder Info")
            f_meta = folders_meta.get(selected_folder)
            if f_meta: st.json(f_meta)
            else: st.warning("No metadata.")
            st.divider()
            if selected_file:
                st.subheader("Document Info")
                fk = os.path.splitext(selected_file)[0]
                i_meta = items_meta.get(fk)
                if i_meta:
                    smart_title = get_smart_title(i_meta, fk)
                    st.markdown(f"**Title:** {smart_title}")
                    if "summary" in i_meta: st.info(i_meta['summary'])
                    with st.expander("Full JSON"): st.json(i_meta)
                    if "ocr" in i_meta and i_meta["ocr"]: 
                        with st.expander("OCR"): st.text(i_meta["ocr"][0])
                else: st.warning("No metadata.")
        with c2:
            if selected_file:
                st.subheader("PDF Preview")
                path = os.path.join(BASE_DIR_FILES, selected_box, selected_folder, selected_file)
                display_pdf(path)
    else:
        st.info("Select items in sidebar.")

# ==========================================
# MÓDULO 2: VISUALIZADOR DE TÓPICOS
# ==========================================
elif app_mode == "🔍 Topic Viewer":
    st.title("🔍 Research Topics & Qrels")
    # ... (Código mantido, pode ser simplificado se necessário) ...
    ecf_data = load_ecf_data()
    all_topics = {}
    if "ExperimentSets" in ecf_data:
        for es in ecf_data["ExperimentSets"]:
            if "Topics" in es: all_topics.update(es["Topics"])
            
    q_docs = load_qrels_data(PATH_QRELS_DOCS)
    q_folders = load_qrels_data(PATH_QRELS_FOLDERS)
    q_boxes = load_qrels_data(PATH_QRELS_BOXES)
    
    topic_ids = list(all_topics.keys())
    topic_ids.sort()
    sel_topic = st.sidebar.selectbox("Topic ID:", topic_ids)
    
    if sel_topic:
        t_data = all_topics[sel_topic]
        st.header(f"{sel_topic}: {t_data.get('TITLE','')}")
        c1, c2 = st.columns(2)
        c1.info(f"**Desc:** {t_data.get('DESCRIPTION','')}")
        c2.warning(f"**Narrative:** {t_data.get('NARRATIVE','')}")
        st.divider()
        st.subheader("Relevant Items (Ground Truth)")
        rd = q_docs.get(sel_topic, [])
        rf = q_folders.get(sel_topic, [])
        rb = q_boxes.get(sel_topic, [])
        t1, t2, t3 = st.tabs([f"Docs ({len(rd)})", f"Folders ({len(rf)})", f"Boxes ({len(rb)})"])
        with t1:
            for did, sc in rd:
                star = "⭐⭐⭐" if sc==3 else "⭐"
                meta = items_meta.get(did, {})
                tit = get_smart_title(meta, did)
                st.markdown(f"**{did}** {star} - {tit}")
                st.markdown("---")
        with t2:
            for fid, sc in rf:
                star = "⭐⭐⭐" if sc==3 else "⭐"
                meta = folders_meta.get(fid, {})
                lbl = meta.get('label') or fid
                st.markdown(f"**{fid}** {star} - {lbl}")
                st.markdown("---")
        with t3:
            for bid, sc in rb:
                star = "⭐⭐⭐" if sc==3 else "⭐"
                st.markdown(f"**{bid}** {star}")

# ==========================================
# MÓDULO 3: RESULTADOS DE EXPERIMENTOS
# ==========================================
elif app_mode == "📊 Experiment Results":
    st.title("📊 Experiment Results Visualization")

    ecf_data = load_ecf_data()
    run_results = load_run_results()
    metrics_data = load_topics_metrics()
    
    # 1. Carrega AMBOS os qrels
    qrels_docs = load_qrels_data(PATH_QRELS_DOCS)
    qrels_folders = load_qrels_data(PATH_QRELS_FOLDERS)

    # 1. Seletor de Experiment Set
    exp_sets = ecf_data.get("ExperimentSets", [])
    if not exp_sets: st.stop()

    set_options = {f"Set {i+1} ({len(es.get('Topics', {}))} Topics)": i for i, es in enumerate(exp_sets)}
    sel_set_label = st.sidebar.selectbox("Select Experiment Set:", list(set_options.keys()))
    sel_set_index = set_options[sel_set_label]
    current_set = exp_sets[sel_set_index]

    with st.spinner("Indexing Training Documents..."):
        training_docs_raw = current_set.get("TrainingDocuments", [])
        # valid_folder_ids agora também é retornado
        valid_doc_ids, valid_folder_ids, docs_by_folder, doc_path_map = parse_training_docs(training_docs_raw)

    # 2. Seletor de Tópico
    set_topics = current_set.get("Topics", {})
    topic_ids = list(set_topics.keys())
    topic_ids.sort()
    sel_topic = st.sidebar.selectbox("Select Topic:", topic_ids)
    
    if sel_topic:
        topic_info = set_topics[sel_topic]
        st.markdown(f"### 📌 {sel_topic}: {topic_info.get('TITLE', '')}")

        # Métricas
        t_metrics = metrics_data.get(sel_topic, {})
        if t_metrics:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("MAP", f"{t_metrics.get('map', 0):.4f}")
            col_m2.metric("nDCG@5", f"{t_metrics.get('ndcg_cut_5', 0):.4f}")
            col_m3.metric("Success@1", f"{t_metrics.get('success_1', 0):.4f}")
            st.markdown("---")

        with st.expander("See Topic Description & Narrative"):
            st.info(topic_info.get('DESCRIPTION'))
            st.warning(topic_info.get('NARRATIVE'))

        st.divider()

        # Preparar dados de relevância DOCS
        topic_qrels_docs = qrels_docs.get(sel_topic, [])
        topic_qrels_docs_map = {did: score for did, score in topic_qrels_docs}

        col_res, col_rel = st.columns(2)

        # --- LADO ESQUERDO: RESULTADOS DO SISTEMA ---
        with col_res:
            st.header("🤖 System Top 5 Folders")
            if sel_topic in run_results:
                folder_scores = run_results[sel_topic]
                sorted_folders = sorted(folder_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for rank, (fid, score) in enumerate(sorted_folders, 1):
                    folder_docs = docs_by_folder.get(fid, [])
                    # Relevância de docs na pasta
                    relevant_in_folder = [
                        (d['doc_id'], topic_qrels_docs_map[d['doc_id']]) 
                        for d in folder_docs if d['doc_id'] in topic_qrels_docs_map
                    ]
                    
                    with st.container(border=True):
                        f_meta = folders_meta.get(fid, {})
                        label = f_meta.get('label', 'Unknown Label')
                        st.markdown(f"**#{rank} Folder: {fid}** (Score: {score:.4f})")
                        st.caption(f"📁 {label}")
                        
                        if relevant_in_folder:
                            count_rel = len(relevant_in_folder)
                            st.markdown(f"🌟 **Contains {count_rel} Relevant Document(s)**")
                        
                        if folder_docs:
                            doc_id_list = [d['doc_id'] for d in folder_docs]
                            st.markdown(f"**Docs:** `{', '.join(doc_id_list)}`")
                            with st.expander(f"Explore Documents ({len(folder_docs)})"):
                                for doc_entry in folder_docs:
                                    did = doc_entry['doc_id']
                                    path = doc_entry['path']
                                    d_meta = items_meta.get(did, {})
                                    smart_title = get_smart_title(d_meta, did)
                                    rel_score = topic_qrels_docs_map.get(did, 0)
                                    rel_badge = ""
                                    if rel_score == 3: rel_badge = "⭐⭐⭐ (High)"
                                    elif rel_score == 1: rel_badge = "⭐ (Relevant)"
                                    
                                    st.markdown(f"📄 **{did}** {rel_badge}")
                                    st.caption(f"{smart_title}")
                                    if st.button("View Details", key=f"det_l_{sel_topic}_{fid}_{did}"):
                                        st.write(f"**Summary:** {d_meta.get('summary', 'No summary')}")
                                        st.write(f"**OCR Start:** {d_meta.get('ocr', ['No OCR'])[0][:500]}...")
                                    if st.checkbox("Show PDF", key=f"pdf_l_{sel_topic}_{fid}_{did}"):
                                        full_path = os.path.join(BASE_DIR_FILES, path)
                                        display_pdf(full_path)
                                    st.markdown("---")
                        else:
                            st.caption("No docs from this folder in current Experiment Set.")
            else:
                st.warning("No run results for this topic.")

        # --- LADO DIREITO: RELEVANTES (CORRIGIDO PARA USAR QRELS DE PASTA) ---
        with col_rel:
            st.header("🎯 Relevant Items")
            
            filter_mode = st.radio(
                "Scope:", 
                ["Training Set Only", "All Qrels (Global)"],
                horizontal=True,
                index=0
            )

            # 1. Contagem de PASTAS (Usando qrels_folders)
            # ---------------------------------------------
            topic_qrels_folders = qrels_folders.get(sel_topic, [])
            
            count_folders_high = 0
            count_folders_rel = 0
            
            for fid, score in topic_qrels_folders:
                # Se for "Training Only", só conta se a pasta estiver no set valid_folder_ids
                if filter_mode == "Training Set Only" and fid not in valid_folder_ids:
                    continue
                
                if score == 3: count_folders_high += 1
                elif score == 1: count_folders_rel += 1

            col_c1, col_c2 = st.columns(2)
            col_c1.metric("Folders w/ High Rel", count_folders_high)
            col_c2.metric("Folders w/ Rel", count_folders_rel)
            
            st.divider()

            # 2. Listagem de DOCUMENTOS (Usando qrels_docs)
            # ---------------------------------------------
            st.subheader("Relevant Documents List")
            sorted_docs_qrels = sorted(topic_qrels_docs, key=lambda x: x[1], reverse=True)
            
            docs_displayed = 0
            if sorted_docs_qrels:
                for did, score in sorted_docs_qrels:
                    is_in_training = did in valid_doc_ids
                    
                    if filter_mode == "Training Set Only" and not is_in_training:
                        continue
                    
                    docs_displayed += 1
                    star = "⭐⭐⭐ High" if score == 3 else "⭐ Relevant"
                    
                    with st.container(border=True):
                        d_meta = items_meta.get(did, {})
                        smart_title = get_smart_title(d_meta, did)
                        
                        st.markdown(f"**{did}** - {star}")
                        st.markdown(f"_{smart_title}_")
                        
                        if not is_in_training:
                            st.warning("🚫 Not in Training Set")
                        
                        with st.expander("View Details & PDF"):
                            st.markdown("**Summary:**")
                            st.info(d_meta.get('summary', 'No summary available.'))
                            
                            pdf_path = None
                            if is_in_training and did in doc_path_map:
                                pdf_path = doc_path_map[did]
                            elif not is_in_training:
                                box = d_meta.get('Sushi Box')
                                folder = d_meta.get('Sushi Folder')
                                file = d_meta.get('Sushi File') or f"{did}.pdf"
                                if box and folder:
                                    pdf_path = os.path.join(box, folder, file)
                            
                            if pdf_path:
                                if st.checkbox("Show PDF", key=f"pdf_r_{sel_topic}_{did}"):
                                    full_path = os.path.join(BASE_DIR_FILES, pdf_path)
                                    display_pdf(full_path)
                            else:
                                st.warning("PDF path could not be resolved.")
            
            if docs_displayed == 0:
                st.info("No documents found for this filter.")