import os
import re
import sys
import time
import shutil
import warnings
from abc import ABC, abstractmethod

import pandas as pd
import pyterrier as pt
from sentence_transformers import SentenceTransformer, util
from pylate import indexes, models, retrieve
import torch

# --- Configuration Constants ---
BM_25_FIELD_WEIGHTS = {
    'title':       {'index_col': 'title',       'w': 2.3,  'c': 0.65},
    'ocr':         {'index_col': 'ocr',         'w': 0.1,  'c': 0.4},
    'folderlabel': {'index_col': 'folderlabel', 'w': 0.2,  'c': 0.6},
    'summary':     {'index_col': 'summary',     'w': 0.08, 'c': 2.0}
}

def get_best_device():
    # 1. Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        return "cuda"
    
    # 2. Check for Apple Silicon GPU (MPS)
    elif torch.backends.mps.is_available():
        return "mps"
    
    # 3. Fallback to CPU
    else:
        return "cpu"

class RetrievalModel(ABC):
    """Abstract base class for all retrieval models."""
    
    @abstractmethod
    def train(self, training_data: list):
        """
        Ingests data and builds index.
        training_data: List of dicts. Each dict MUST contain:
                       'docno', 'folder', 'box', 'date', 
                       plus specific fields ('title', 'ocr', etc.)
                       and a 'text_blob' for dense retrievers.
        """
        pass

    @abstractmethod
    def search(self, query: str) -> pd.DataFrame:
        """
        Returns DataFrame with columns ['folder', 'score', 'docno']
        """
        pass

class BM25Model(RetrievalModel):
    def __init__(self, searching_fields):
        self.searching_fields = searching_fields
        self.retriever = None
        self._init_pyterrier()

    def _init_pyterrier(self):
        if not pt.java.started():
            if os.name == 'nt': 
                # Ideally, pass this via env var, but keeping your logic:
                os.environ["JAVA_HOME"] = r'C:\Program Files\Java\jdk-11'
            pt.java.init()
        pt.ApplicationSetup.setProperty("terrier.use.memory.mapping", "false")
        pt.java.set_log_level('ERROR')

    def train(self, training_data):
        # 1. Determine Weights (BM25 vs BM25F)
        # Handle the nested list structure from your config (e.g. [['title']])
        current_fields = self.searching_fields[0] if isinstance(self.searching_fields[0], list) else self.searching_fields
        
        active_text_attrs = []
        #controls = {}
        wmodel = "BM25"

        if len(current_fields) > 1:
            wmodel = "BM25F"
            for idx, field in enumerate(current_fields):
                if field in BM_25_FIELD_WEIGHTS:
                    col_name = BM_25_FIELD_WEIGHTS[field]['index_col']
                    active_text_attrs.append(col_name)
                    #controls[f'w.{idx}'] = BM_25_FIELD_WEIGHTS[field]['w']
                    #controls[f'c.{idx}'] = BM_25_FIELD_WEIGHTS[field]['c']
        else:
            # Simple BM25 on the specific field
            active_text_attrs = [BM_25_FIELD_WEIGHTS[f]['index_col'] for f in current_fields if f in BM_25_FIELD_WEIGHTS]

        # 2. Indexing
        index_dir = os.path.abspath(os.path.join("terrierindex", str(int(time.time()))))
        indexer = pt.IterDictIndexer(
            index_dir, 
            meta={'docno': 20, 'folder': 20, 'box': 20, 'date': 10}, 
            text_attrs=active_text_attrs,
            meta_reverse=['docno'], 
            overwrite=True,
            fields=(wmodel == "BM25F")
        )
        indexref = indexer.index(training_data)
        index = pt.IndexFactory.of(indexref)

        # 3. Retriever Setup
        self.retriever = pt.terrier.Retriever(
            index, 
            wmodel=wmodel,
            #controls=controls, 
            metadata=['docno', 'folder', 'box', 'date'], 
            num_results=1000
        )

    def search(self, query):
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        if not clean_query.strip(): 
            return pd.DataFrame()
        
        result = self.retriever.search(clean_query)
        # Ensure we return standard columns
        return result[['folder', 'score', 'docno']]

class EmbeddingsModel(RetrievalModel):
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name, device=get_best_device())
        self.doc_embeddings = None
        self.metadata_map = [] # List to store metadata by index

    def train(self, training_data):
        texts = []
        self.metadata_map = []
        
        for doc in training_data:
            # Use the pre-computed text blob
            texts.append(doc['text_blob'])
            self.metadata_map.append({
                'docno': doc['docno'],
                'folder': doc['folder']
            })

        self.doc_embeddings = self.model.encode(texts, convert_to_tensor=True)

    def search(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        # Move to CPU and list
        scores = cosine_scores.tolist()
        
        results = []
        for idx, score in enumerate(scores):
            results.append({
                'docno': self.metadata_map[idx]['docno'],
                'folder': self.metadata_map[idx]['folder'],
                'score': score
            })
            
        df = pd.DataFrame(results)
        return df.sort_values(by='score', ascending=False)

class ColBERTModel(RetrievalModel):
    def __init__(self, index_path="pylate-index"):
        self.index_path = index_path
        self.colbert_model = models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0", device=get_best_device())
        self.colbert_retriever = None
        self.doc_map = {} # Maps docid -> folder

    def train(self, training_data):
        # Clean previous index to prevent corruption on re-runs
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)

        colbert_index = indexes.PLAID(
            index_folder=self.index_path,
            index_name="index",
            override=True,
        )
        self.colbert_retriever = retrieve.ColBERT(index=colbert_index)

        texts = []
        ids = []
        self.doc_map = {}

        for doc in training_data:
            texts.append(doc['text_blob'])
            ids.append(str(doc['docno']))
            self.doc_map[str(doc['docno'])] = doc['folder']

        doc_embeddings = self.colbert_model.encode(
            texts,
            batch_size=128,
            is_query=False,
            show_progress_bar=False,
        )

        colbert_index.add_documents(
            documents_ids=ids,
            documents_embeddings=doc_embeddings,
        )

    def search(self, query):
        query_embeddings = self.colbert_model.encode(
            [query],
            batch_size=128,
            is_query=True,
            show_progress_bar=False,
        )

        results = self.colbert_retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=100, # or 1000
        )
        
        # Results is list of list. Get first query results.
        data = []
        for item in results[0]:
            doc_id = str(item['id'])
            data.append({
                'docno': doc_id,
                'folder': self.doc_map.get(doc_id, "Unknown"),
                'score': item['score']
            })
            
        return pd.DataFrame(data)