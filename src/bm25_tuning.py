# About the Tuning Params:
# 1. The BM25F parameters are not a question about any particular ECF, so you don't want to start with the ECF that you will use in test -- a person making a BM25F system would not know the specific ECF's that would be user at test time.  
# 2. The second thing to say is that BM25 includes a document independence assumption, which means that we first compute a score for each document and then rank the documents. So the training mask does not change the scores of the selected documents.  

# **"Unfair Optimization:"**
# "I would suggest just doing the unfair optimization on the entire topic set first, and then circling back to see if we have time to add cross-validation at the end."
    # **Implementation:** This suggests a really simple design for BM25F parameter tuning in which you use no training mask (the all-documents condition).
    #   - Now you can easily optimize without cross-validation by just doing a grid search in the parameter space.
    #       * BM25F has 8 parameters for four fields, so you will probably want something more efficient than a grid search. 
    #   - And you'll have a lot of ties (because of the small number of relevant documents) so you want something that finds midpoints in the parameter space for tied maximal values -- this is a sort of maxent approach, but on quantized results (so its not really maxent).  
    #   - I suggest to you do some by-hand eploration of the parameter space (one dimension at a time) to see this quantization effect.  
    #   - Choose an evaluation measure to optimize for: nDCG@10 might be better than nDCG@5, since it would have fewer ties.

# **Cross-Validation:**
    # - To implement cross-validation, choose some test topics and optimize the BM25F parameters on the other topics. 
    # - Then whenever you run an actual experiment whenever you test on one of those test topics, just use the BM25 parameters that were learned on the other topic.  

import os
import json
import itertools
import pandas as pd
import pyterrier as pt
import pytrec_eval
from tqdm import tqdm

# Setup PyTerrier
if not pt.java.started():
    if os.name == 'nt':
        os.environ["JAVA_HOME"] = r'C:\Program Files\Java\jdk-11'
    pt.java.init()
pt.java.set_log_level('ERROR')

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Adjust these paths to match your project structure
ITEMS_METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'items_metadata', 'itemsV1.2.json')
FOLDER_METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'folders_metadata', 'FoldersV1.3.json')
FOLDER_QRELS_PATH = os.path.join(PROJECT_ROOT, 'qrels', 'formal-run-qrels', 'formal-folder-qrel.txt')
TOPICS_PATH = os.path.join(PROJECT_ROOT, "src", "data_creation", "topics_output.txt")
ECF_PATH = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated', 'ECF_ALL_TRAINING_SET.json')

# Define your parameter grid here
# w = weight, c = saturation
PARAM_GRID = {
    'title':       {'w': [1.0, 2.3, 5.0],  'c': [0.5, 0.65, 0.8]},
    'ocr':         {'w': [0.1, 0.5, 1.0],  'c': [0.1, 0.4, 10.0]},
    'folderlabel': {'w': [0.2, 1.0],       'c': [0.6, 1.0]},
    'summary':     {'w': [0.08, 0.5],      'c': [2.0, 5.0]}
}

class BM25FTuner:
    def __init__(self):
        self.load_data()
        self.qrels = self.load_qrels()
        
        # Define fields order strictly to match PyTerrier's w.0, w.1 index mapping
        self.fields = ['title', 'folderlabel', 'ocr', 'summary']
        
    def load_data(self):
        """Loads metadata and topics."""
        with open(ITEMS_METADATA_PATH) as f: self.items = json.load(f)
        with open(FOLDER_METADATA_PATH) as f: self.folder_metadata = json.load(f)
        with open(ECF_PATH) as f: self.ecf = json.load(f)
        
        # Load Topics
        with open(TOPICS_PATH, 'r', encoding='utf-8') as f:
            topics_raw = json.load(f)
        
        # Convert to DataFrame for PyTerrier
        self.topics_df = pd.DataFrame([
            {'qid': t['ID'], 'query': t['TITLE']} 
            for t in topics_raw.values()
        ])

    def load_qrels(self):
        """Loads QRELs into a dictionary for pytrec_eval."""
        qrels = {}
        with open(FOLDER_QRELS_PATH) as f:
            for line in f:
                qid, _, docno, label = line.split('\t')
                if qid not in qrels: qrels[qid] = {}
                qrels[qid][docno] = int(label.strip())
        return qrels

    def prepare_training_data(self):
        """Prepares the generator for indexing."""
        for trainingDoc in self.ecf["ExperimentSets"][0]["TrainingDocuments"]:
            file = trainingDoc[-10:-4]
            if file not in self.items: continue
            
            folder = self.items[file]['Sushi Folder']
            
            # Base data
            doc_entry = {
                'docno': file,
                'folder': folder, # Stored for retrieval mapping
                'title': self.items[file]['title'],
                'ocr': self.items[file]['ocr'][0] if self.items[file]['ocr'] else "",
                'summary': self.items[file]['summary'],
            }

            # Folder Label Logic
            try:
                meta = self.folder_metadata[folder]
                label = f"{meta.get('label_parent_expanded', '')}. {meta.get('scope_stoppers', '')}"
            except:
                label = self.folder_metadata.get(folder, {}).get('label', '')
            doc_entry['folderlabel'] = label

            yield doc_entry

    def index_collection(self):
        """Creates the Terrier index once with all fields."""
        print(">>> Indexing Collection...")
        index_dir = "./tuning_index"
        
        indexer = pt.IterDictIndexer(
            index_dir,
            meta={'docno': 20, 'folder': 20},
            text_attrs=self.fields, # Index these specific fields
            overwrite=True
        )
        
        indexref = indexer.index(self.prepare_training_data())
        self.index = pt.IndexFactory.of(indexref)
        print(">>> Indexing Complete.")

    def run_grid_search(self):
        """Iterates through parameter combinations and evaluates."""
        
        # 1. Generate all combinations
        # We need a list of (w, c) tuples for each field
        keys = self.fields 
        value_lists = []
        
        for field in keys:
            # Create pairs of (w, c) for this field
            # e.g., [(1.0, 0.5), (1.0, 0.65), (2.3, 0.5)...]
            pairs = list(itertools.product(PARAM_GRID[field]['w'], PARAM_GRID[field]['c']))
            value_lists.append(pairs)
        
        # Cartesian product of all field configurations
        # combination structure: ((w_title, c_title), (w_folder, c_folder), ...)
        combinations = list(itertools.product(*value_lists))
        
        print(f">>> Starting Grid Search with {len(combinations)} combinations...")
        
        results = []

        # 2. Iterate
        for comb in tqdm(combinations, desc="Tuning"):
            controls = {}
            params_log = {}
            
            # Map combination back to PyTerrier controls (w.0, c.0, w.1, c.1...)
            for i, (w, c) in enumerate(comb):
                controls[f'w.{i}'] = w
                controls[f'c.{i}'] = c
                
                # For logging results later
                field_name = self.fields[i]
                params_log[f'{field_name}_w'] = w
                params_log[f'{field_name}_c'] = c

            # 3. Retrieval
            # We use BM25F model with dynamic controls
            bm25f = pt.BatchRetrieve(
                self.index, 
                wmodel="BM25F", 
                controls=controls,
                metadata=['docno', 'folder'],
                num_results=100
            )

            res = bm25f.transform(self.topics_df)

            # 4. Evaluation (In-Memory)
            # Convert PyTerrier result to pytrec_eval format
            # Result needs to be mapped to Folder ID if your QRELs are folder-based
            run_dict = {}
            for _, row in res.iterrows():
                qid = str(row['qid'])
                folder = row['folder']
                score = float(row['score'])
                
                if qid not in run_dict: run_dict[qid] = {}
                # Handle multiple docs per folder: keep max score for that folder
                if folder not in run_dict[qid] or score > run_dict[qid][folder]:
                    run_dict[qid][folder] = score

            evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, {'ndcg_cut_10'})
            metrics = evaluator.evaluate(run_dict)

            # Calculate Mean nDCG@10
            ndcg_scores = [m['ndcg_cut_10'] for m in metrics.values()]
            mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

            # Store result
            entry = {**params_log, 'ndcg_10': mean_ndcg}
            results.append(entry)

        # 5. Output Results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='ndcg_10', ascending=False)
        
        print("\n>>> Top 5 Configurations:")
        print(results_df.head(5))
        
        # Save to CSV
        results_df.to_csv("bm25f_tuning_results.csv", index=False)
        print(">>> Full results saved to bm25f_tuning_results.csv")

if __name__ == "__main__":
    tuner = BM25FTuner()
    tuner.index_collection()
    tuner.run_grid_search()