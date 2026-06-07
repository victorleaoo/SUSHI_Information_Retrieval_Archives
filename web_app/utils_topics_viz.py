import os
import json
import base64
import streamlit as st

current_dir = os.path.dirname(__file__)

PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '..'))
PATH_ECF = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated', 'ECF_ALL_TRAINING_SET.json')
BASE_DIR_FILES = os.path.join(PROJECT_ROOT, 'data', 'raw')
PATH_FOLDERS_JSON = os.path.join(PROJECT_ROOT, 'data', 'folders_metadata', 'FoldersV1.3.json')
PATH_ITEMS_JSON = os.path.join(PROJECT_ROOT, 'data', 'items_metadata', 'itemsV1.2.json')
PATH_TOPICS_JSON = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated', 'ECF_ALL_TRAINING_SET.json')
PATH_QRELS_DOCS = os.path.join(PROJECT_ROOT, 'qrels', 'formal-document-qrel.txt')
PATH_QRELS_FOLDERS = os.path.join(PROJECT_ROOT, 'qrels', 'formal-folder-qrel.txt')
PATH_QRELS_BOXES = os.path.join(PROJECT_ROOT, 'qrels', 'formal-box-qrel.txt')

@st.cache_data
def load_metadata():
    """Load folders and items metadata JSON files."""
    folders_data, items_data = {}, {}
    try:
        with open(PATH_FOLDERS_JSON, 'r', encoding='utf-8') as f:
            folders_data = json.load(f)
    except FileNotFoundError:
        print('folders_data file not found')
        pass
    try:
        with open(PATH_ITEMS_JSON, 'r', encoding='utf-8') as f:
            items_data = json.load(f)
    except FileNotFoundError:
        print('items_data file not found')
        pass
    return folders_data, items_data

@st.cache_data
def load_ecf_data():
    """Load ECF training set JSON data."""
    try:
        with open(PATH_ECF, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print('ecf file not found')
        return {}

@st.cache_data
def load_qrels_data(filepath):
    """Load qrels data from a file path into a topic->[(id, relevance)] map."""
    qrels = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    topic_id = parts[0]
                    raw_id = parts[2]
                    item_id = raw_id[:-4] if raw_id.lower().endswith('.pdf') else raw_id
                    try: relevance = int(parts[3])
                    except ValueError: continue
                    if relevance > 0:
                        if topic_id not in qrels: qrels[topic_id] = []
                        qrels[topic_id].append((item_id, relevance))
    except FileNotFoundError:
        return {}
    return qrels

def get_smart_title(item_data, item_id):
    """Derive a readable title from metadata, falling back to the item id."""
    if not item_data: return item_id
    brown_title = item_data.get("Brown Title")
    nara_title = item_data.get("NARA Title")
    if brown_title and not isinstance(brown_title, float): return brown_title
    if nara_title and not isinstance(nara_title, float):
        start = nara_title.find('Concerning')
        if start != -1: nara_title = nara_title[start+11:]
        end1 = nara_title.rfind(':')
        end2 = nara_title.rfind('(')
        end = min(end1, end2) if end1 != -1 and end2 != -1 else max(end1, end2)
        if end != -1: nara_title = nara_title[:end]
        return nara_title
    return item_id

def get_file_path_from_metadata(doc_id, items_meta, folders_meta):
    """Build the PDF file path from item and folder metadata."""
    item_data = items_meta.get(doc_id)
    if not item_data: return None
    folder_name = item_data.get('Sushi Folder')
    if not folder_name: return None
    box_name = item_data.get('Sushi Box') 
    if box_name and folder_name:
        return os.path.join(BASE_DIR_FILES, box_name, folder_name, doc_id + ".pdf")
    return None

def get_pdf_base64(file_path):
    """Return base64-encoded PDF content for a file path, or None on failure."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            return None
    return None