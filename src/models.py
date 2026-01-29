import os
import re
import time
import shutil
from abc import ABC, abstractmethod

import pandas as pd
import pyterrier as pt
from sentence_transformers import SentenceTransformer, util
from pylate import indexes, models, retrieve
import torch

# --- Configuration Constants ---
## TOFS Tuned
BM_25_FIELD_WEIGHTS = {
    'title':       {'index_col': 'title',       'w': 3.0,  'c': 0.5},
    'ocr':         {'index_col': 'ocr',         'w': 0.5,  'c': 0.4},
    'folderlabel': {'index_col': 'folderlabel', 'w': 1.3,  'c': 0.65},
    'summary':     {'index_col': 'summary',     'w': 1.0, 'c': 1.5}
}

## TFS Tuned
# BM_25_FIELD_WEIGHTS = {
#     'title':       {'index_col': 'title',       'w': 1.0,  'c': 0.5},
#     'folderlabel': {'index_col': 'folderlabel', 'w': 1.3,  'c': 0.3},
#     'summary':     {'index_col': 'summary',     'w': 1.0, 'c': 0.85}
# }

def get_best_device():
    """
    Detects the best available hardware accelerator for PyTorch operations.
    
    Priority order:
    1. NVIDIA GPU (CUDA)
    2. Apple Silicon GPU (MPS)
    3. CPU (Fallback)
    """
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
    """
    Abstract base class defining the contract for all retrieval algorithms.
    
    Ensures that any new model (Sparse, Dense, or Hybrid) implements the standard `train` (indexing) and `search` (retrieval) methods required by the experiment pipeline.
    """
    
    @abstractmethod
    def train(self, training_data: list):
        """
        Ingests the dataset and builds the necessary search index.
        
        Args:
            training_data (list[dict]): A list of document dictionaries. 
                Each dictionary must contain standard metadata:
                - 'docno': Unique document identifier.
                - 'folder': The folder ID the document belongs to.
                - 'box', 'date': Additional metadata.
                - specific fields ('title', 'ocr', etc.) based on configuration.
                - 'text_blob': A pre-concatenated string for dense retrievers.
        """
        pass

    @abstractmethod
    def search(self, query: str) -> pd.DataFrame:
        """
        Performs a search query against the built index.
            
        Returns:
            pd.DataFrame: A DataFrame containing at least these columns:
                - 'docno': The retrieved document ID.
                - 'folder': The folder ID associated with the document.
                - 'score': The relevance score computed by the model.
        """
        pass

class BM25Model(RetrievalModel):
    """
    Wrapper for PyTerrier's BM25 and BM25F implementations.
    
    Automatically switches between standard BM25 (single field) and BM25F (multifield) based on the number of searching fields provided.
    """
    def __init__(self, 
                 searching_fields):
        """
        Initializes the BM25/BM25F model configuration.

        Args:
            searching_fields (list): List of fields to index (e.g., ['title', 'ocr']).
        """
        self.searching_fields = searching_fields
        self.retriever = None
        self._init_pyterrier()

    def _init_pyterrier(self):
        """
        Initializes the Java Virtual Machine (JVM) for PyTerrier.
        
        Sets necessary environment variables (like JAVA_HOME on Windows) and configures PyTerrier settings to avoid memory mapping issues.
        """
        if not pt.java.started():
            if os.name == 'nt': 
                os.environ["JAVA_HOME"] = r'C:\Program Files\Java\jdk-11'
            pt.java.init()
        pt.ApplicationSetup.setProperty("terrier.use.memory.mapping", "false")
        pt.java.set_log_level('ERROR')

    def train(self, training_data):
        """
        Builds a PyTerrier index from the training data.
        
        Logic:
        1. Checks `self.searching_fields`.
        2. If > 1 field is used, it configures **BM25F** using weights `w` and saturation params `c` defined in `BM_25_FIELD_WEIGHTS`.
        3. If 1 field is used, it configures standard **BM25**.
        4. Creates an IterDictIndexer to build the index on disk.
        """
        # 1. Determine Weights (BM25 vs BM25F)
        current_fields = self.searching_fields
        
        active_text_attrs = []
        controls = {}
        wmodel = "BM25"

        if len(current_fields) > 1:
            wmodel = "BM25F"
            for idx, field in enumerate(current_fields):
                if field in BM_25_FIELD_WEIGHTS:
                    col_name = BM_25_FIELD_WEIGHTS[field]['index_col']
                    active_text_attrs.append(col_name)
                    # Map weights to PyTerrier controls (w.0, c.0, etc.)
                    controls[f'w.{idx}'] = BM_25_FIELD_WEIGHTS[field]['w']
                    controls[f'c.{idx}'] = BM_25_FIELD_WEIGHTS[field]['c']
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
            controls=controls, 
            metadata=['docno', 'folder', 'box', 'date'], 
            num_results=1000
        )

    def search(self, query):
        """
        Cleans the query string and executes retrieval.
        
        Removes special characters to prevent PyTerrier query parser errors.
        Returns a formatted DataFrame with standard columns.
        """
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        if not clean_query.strip(): 
            return pd.DataFrame()
        
        result = self.retriever.search(clean_query)
        return result[['folder', 'score', 'docno']]

class EmbeddingsModel(RetrievalModel):
    """
    Dense Retrieval model using SentenceTransformers (Bi-Encoder).
    
    Encodes all documents into vector embeddings and performs 
    Cosine Similarity search for retrieval.
    """
    def __init__(self, 
                 model_name='all-mpnet-base-v2'):
        """
        Initializes the SentenceTransformer model.
        
        Args:
            model_name (str): HuggingFace model identifier.
        """
        self.model = SentenceTransformer(model_name, 
                                         device=get_best_device())
        self.doc_embeddings = None
        self.metadata_map = []

    def train(self, training_data):
        """
        Encodes the document collection into a tensor matrix.
        
        It expects a 'text_blob' field in `training_data` which contains the concatenated text representation of the document.
        """
        texts = []
        self.metadata_map = []
        
        for doc in training_data:
            texts.append(doc['text_blob'])
            self.metadata_map.append({
                'docno': doc['docno'],
                'folder': doc['folder']
            })

        self.doc_embeddings = self.model.encode(texts, convert_to_tensor=True)

    def search(self, query):
        """
        Encodes the query and calculates Cosine Similarity against all docs.
        
        Returns:
            pd.DataFrame: Ranked results sorted by similarity score (descending).
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
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
    """
    Late Interaction Retrieval model using ColBERT via the `pylate` library.
    
    ColBERT computes embeddings for every token and performs "MaxSim" interaction, offering high precision but higher computational cost.
    Uses PLAID indexing for efficiency.
    """
    def __init__(self, 
                 index_path="pylate-index"):
        self.index_path = index_path
        self.colbert_model = models.ColBERT(model_name_or_path="lightonai/colbertv2.0", 
                                            device=get_best_device())
        self.colbert_retriever = None
        self.doc_map = {} # Maps docid -> folder

    def train(self, training_data):
        """
        Builds a PLAID index for ColBERT.
        
        1. Clears any existing index at `self.index_path`.
        2. Encodes document `text_blob`s using ColBERT.
        3. Adds document embeddings to the index.
        """
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
            batch_size=512,
            is_query=False,
            show_progress_bar=False,
        )

        colbert_index.add_documents(
            documents_ids=ids,
            documents_embeddings=doc_embeddings,
        )

    def search(self, query):
        """
        Retrieves top-k documents using ColBERT interaction.
        
        1. Encodes the query.
        2. Retrieves results from the PLAID index.
        3. Maps internal IDs back to `docno` and `folder`.
        """
        query_embeddings = self.colbert_model.encode(
            [query],
            batch_size=512,
            is_query=True,
            show_progress_bar=False,
        )

        results = self.colbert_retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=100, # or 1000
        )
        
        data = []
        for item in results[0]:
            doc_id = str(item['id'])
            data.append({
                'docno': doc_id,
                'folder': self.doc_map.get(doc_id, "Unknown"),
                'score': item['score']
            })
            
        return pd.DataFrame(data)