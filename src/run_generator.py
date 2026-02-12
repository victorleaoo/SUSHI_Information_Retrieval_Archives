import os
import statistics
import warnings
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from models import BM25Model, EmbeddingsModel, ColBERTModel
from evaluator import Evaluator
from data_loader import DataLoader

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Style:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# CONSTANTS
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'RunResults.tsv')
FOLDER_QRELS_PATH = os.path.join(PROJECT_ROOT, 'qrels', 'formal-folder-qrel.txt')
BOX_QRELS_PATH = os.path.join(PROJECT_ROOT, 'qrels', 'formal-box-qrel.txt')

RANDOM_SEED_LIST = [1, 42, 100, 300, 333, 777, 999, 2025, 6159, 12345, 19865, 53819,
                    56782, 62537, 72738, 75259, 81236, 91823, 98665, 98765, 99009, 999777333,
                    120302, 123865, 170302, 180803, 5122025, 12052024, 12052025, 99000011]

SEARCHING_FIELD_MAP = {'folderlabel': 'F', 'ocr': 'O', 'summary': 'S', 'title': 'T'}
EXPANSION_NAME_MAP = {'same_box': 'SB', 'same_snc': 'SS', 'similar_snc': 'SMS', 'close_date': 'CD'}
RFF_WEIGHTS = {'bm25': 1.0, 'embeddings': 0.65, 'colbert': 0.65}
RRF_R_PARAMETER = 0 

class RunGenerator:
    """
    Orchestrates the entire SUSHI experiment pipeline.

    This class manages:
    1.  **Configuration**: Selection of models, fields, expansion strategies, and sampling.
    2.  **Execution Loop**: Iterating through random seeds to simulate sparse data scenarios.
    3.  **Model Training**: Initializing and training specified retrieval models (BM25, ColBERT, etc.).
    4.  **Retrieval & Fusion**: Executing searches, applying Rank Fusion (RRF), and managing expansion logic.
    5.  **Evaluation**: Triggering the evaluation of results against QRELs.

    Attributes:
        searching_fields (list): List of field combinations to search (e.g., [['title'], ['ocr'], ['title', 'ocr']]).
        query_fields (list): List of query modes to use (e.g., ['T', 'TD']).
        run_type (str): 'random' (for sparse sampling) or 'all_documents' (oracle).
        models (list): List of model names to ensemble (e.g., ['bm25', 'colbert']).
        sampling (str): 'uniform' for uniform sampling or 'uneven' for skewed sampling.
        expansion (list): List of expansion techniques to apply (e.g., ['same_box', 'similar_snc']).
        rrf_input (str): Strategy for fusion ('docs' = Early Fusion, 'folders' = Late Fusion).
        expansion_ceiling_k (int): Rank threshold that expanded results cannot surpass.
    """
    def __init__(self, 
                 searching_fields=[['title', 'ocr', 'folderlabel', 'summary']],
                 query_fields=['TD'],
                 run_type='random',
                 models=['bm25', 'embeddings', 'colbert'],
                 sampling='uniform',
                 expansion=[],
                 all_folders_folder_label=False,
                 rrf_input='docs',
                 expansion_ceiling_k=2
                 ):
        self.searching_fields = searching_fields
        self.query_fields = query_fields
        self.run_type = run_type
        self.models = models
        self.sampling = sampling
        self.expansion = expansion
        self.all_folders_folder_label = all_folders_folder_label
        self.rrf_input = rrf_input
        self.expansion_ceiling_k = expansion_ceiling_k

        self.loader = DataLoader(PROJECT_ROOT)
        self.items = self.loader.items
        self.folderMetadata = self.loader.folder_metadata
        
        self.evaluator = Evaluator(FOLDER_QRELS_PATH, BOX_QRELS_PATH)
    
    def run_experiments(self):
        """
        Main execution loop.
        
        Iterates through all configured field combinations and query types. 
        For 'random' runs, it iterates through a fixed list of random seeds to ensure statistical significance. For 'all_documents', it runs once.
        Results are saved to disk and evaluated immediately.
        """
        for searching_field in self.searching_fields:
            for query_field in self.query_fields:
                # Set current context
                self.current_searching_field = searching_field
                self.current_query_field = query_field
                
                print(f"{Style.BOLD}{Style.GREEN}> Running Experiments for:{Style.RESET}")
                s_fields_str = ', '.join([f for f in searching_field]) if isinstance(searching_field[0], str) else str(searching_field)
                print(f"\t- {Style.BOLD}Searching fields:{Style.RESET}  {Style.CYAN}{s_fields_str}{Style.RESET}")
                print(f"\t- {Style.BOLD}Query fields:{Style.RESET}      {Style.CYAN}{self.current_query_field}{Style.RESET}")
                print(f"\t- {Style.BOLD}Run type:{Style.RESET}          {Style.CYAN}{self.run_type}{Style.RESET}")
                print(f"\t- {Style.BOLD}Model:{Style.RESET}             {Style.CYAN}{', '.join(self.models)}{Style.RESET}")
                print(f"\t- {Style.BOLD}Expansion:{Style.RESET}         {Style.CYAN}{', '.join(self.expansion) if self.expansion else 'None'}{Style.RESET}")
                print(f"\t- {Style.BOLD}RRF Input:{Style.RESET}         {Style.CYAN}{self.rrf_input}{Style.RESET}")

                # Setup Output Directory
                run_folder_name = self.saving_folder_name()
                metrics_output_folder = os.path.abspath(f'../all_runs/{run_folder_name}')
                os.makedirs(metrics_output_folder, exist_ok=True)

                if self.run_type == 'random':
                    for random_seed in tqdm(RANDOM_SEED_LIST, desc=f"Runs ({run_folder_name})"):
                        # 1. Execute Run (Delegated to run_single_seed)
                        results = self.run_single_seed(random_seed, searching_field, query_field)

                        # 2. Save Run File
                        run_name = f'45-Topics-Random-{random_seed}'
                        self.evaluator.save_run_file(results, RESULTS_PATH, run_name)

                        # 3. Evaluate & Save Metrics
                        json_path = os.path.join(metrics_output_folder, f'Random{random_seed}_TopicsFolderMetrics.json')
                        self.evaluator.evaluate(RESULTS_PATH, json_path)

                    # 4. Generate Aggregate Stats (After all seeds are done)
                    self.evaluator.generate_aggregated_metrics(metrics_output_folder, 'random')

                elif self.run_type == 'all_documents':
                    # Single execution
                    results = self.run_single_seed(0, searching_field, query_field)

                    run_name = '45-Topics-AllDocuments'
                    self.evaluator.save_run_file(results, RESULTS_PATH, run_name)

                    json_path = os.path.join(metrics_output_folder, 'AllDocuments_TopicsFolderMetrics.json')
                    self.evaluator.evaluate(RESULTS_PATH, json_path)
                    
                    self.evaluator.generate_aggregated_metrics(metrics_output_folder, 'all_documents')

    def run_single_seed(self, random_seed, searching_field, query_field):
        """
        Executes the full retrieval pipeline for a single random seed.

        Steps:
        1. Generates the ECF (sample of training docs).
        2. Prepares training data.
        3. Builds relations for expansion (if enabled).
        4. Trains all active models.
        5. Generates topic search results.

        Returns:
            list: Ranked results for all topics.
        """
        self.current_searching_field = searching_field
        self.current_query_field = query_field
        self.random_seed = random_seed
        
        # 1. Create/Load ECF via DataLoader
        if self.run_type == 'all_documents':
            self.ecf = self.loader.load_all_docs_ecf()
        else:
            self.ecf = self.loader.create_random_ecf(random_seed, self.sampling)

        # 2. Prepare Data
        clean_data = self.prepare_training_data()

        # Create relations for expansion
        if self.run_type != "all_documents" and self.all_folders_folder_label == False:
            self.relations = self.create_folder_relations_for_expansion(clean_data)

        # 3. Train Models
        self.active_models = {}
        for model_name in self.models:
            if model_name == 'bm25':
                model = BM25Model(self.current_searching_field)
            elif model_name == 'embeddings':
                model = EmbeddingsModel()
            elif model_name == 'colbert':
                model = ColBERTModel()

            model.train(clean_data)
            self.active_models[model_name] = model
        
        # 4. Generate Results
        results = self.produce_topics_results()
        return results

    def prepare_training_data(self):
        """
        Formats raw metadata into a list of training dictionaries for the models.
        
        Handles:
        - Text concatenation for dense models ('text_blob').
        - Field selection based on configuration.
        - Special handling for 'ALLFL' (All Folders Label) mode.
        """
        trainingSet = []
        current_fields = self.current_searching_field

        # If ALLFL is True, it uses a folder metadata label approach only
        if self.all_folders_folder_label:
             for folder in self.folderMetadata:
                try:
                    label = self.folderMetadata[folder]['label_parent_expanded']
                except:
                    label = self.folderMetadata[folder]['label']
                
                trainingSet.append({
                    'docno': folder,
                    'folder': folder,
                    'box': self.folderMetadata[folder]['box'],
                    'date': self.folderMetadata[folder]['date'],
                    'folderlabel': label,
                    'text_blob': label 
                })
        else:
            # Standard Document-level Training
            for trainingDoc in self.ecf["ExperimentSets"][0]["TrainingDocuments"]: 
                file = trainingDoc[-10:-4] 
                folder = self.items[file]['Sushi Folder']
                
                doc_entry = {
                    'docno': file,
                    'folder': folder,
                    'box': self.items[file]['Sushi Box'],
                    'date': self.items[file]['date'],
                    'title': self.items[file]['title'],
                    'ocr': self.items[file]['ocr'][0],
                    'summary': self.items[file]['summary'],
                }

                try:
                    label = self.folderMetadata[folder]['label_parent_expanded']
                except:
                    label = self.folderMetadata[folder]['label']
                doc_entry['folderlabel'] = label

                text_blob = ""
                
                for field in current_fields:
                    if field == 'folderlabel':
                        text_blob += label + ". "
                    else:
                        val = self.items[file][field]
                        if field == 'ocr': val = val[0]
                        text_blob += str(val) + ". "
                doc_entry['text_blob'] = text_blob.strip()
                trainingSet.append(doc_entry)

        return trainingSet
    
    def produce_topics_results(self):
        """
        Runs search for all topics defined in the ECF.
        
        Handles:
        - Query construction based on topic fields (Title, Description).
        - Executing search on all active models.
        - Routing between Early Fusion ('docs') and Late Fusion ('folders').
        - Triggering Expansion logic based on configuration.
        """
        results = []
        topics = list(self.ecf['ExperimentSets'][0]['Topics'].keys())
        
        i = 0
        for j in range(len(topics)):
            results.append({})
            results[i]['Id'] = topics[j]

            title = self.ecf['ExperimentSets'][0]['Topics'][topics[j]].get('TITLE', '')
            description = self.ecf['ExperimentSets'][0]['Topics'][topics[j]].get('DESCRIPTION', '')
            narrative = self.ecf['ExperimentSets'][0]['Topics'][topics[j]].get('NARRATIVE', '')

            if self.current_query_field == "TDN":
                query = f"{title} {description} {narrative}".strip()
            elif self.current_query_field == "TD":
                query = f"{title}. {description}".strip()
            else:
                query = title.strip()

            # 1. Get Raw Results from all models
            raw_results_map = {}
            for model_name, model_instance in self.active_models.items():
                raw_results_map[model_name] = model_instance.search(query)

            # 2. Pipeline Logic Branching
            if self.rrf_input == 'folders':
                expanded_map = {}
                for model_name, raw_df in raw_results_map.items():
                    # Check if it should expand or just take raw scores
                    if len(self.expansion) > 0 and self.run_type != 'all_documents':
                        expanded_map[model_name] = self.produce_expansion_results(raw_df)
                    else:
                        expanded_map[model_name] = raw_df[['folder', 'score']]
                
                if len(self.models) > 1:
                    final_ranked_df = self.apply_folder_level_rrf(expanded_map)
                else:
                    final_ranked_df = expanded_map[self.models[0]]
            
            elif self.rrf_input == 'docs':
                if len(self.models) > 1:
                    fused_docs_df = self.apply_document_level_rrf(raw_results_map)
                else:
                    fused_docs_df = raw_results_map[self.models[0]]
                
                # Expand the fused list
                if len(self.expansion) > 0 and self.run_type != 'all_documents':
                    final_ranked_df = self.produce_expansion_results(fused_docs_df)
                else:
                    # If no expansion, just aggregate doc scores to folders
                    final_ranked_df = fused_docs_df.groupby('folder', as_index=False)['score'].max()

            # 3. Sort and Format
            if 'score' in final_ranked_df.columns:
                final_ranked_df = final_ranked_df.sort_values('score', ascending=False)
            elif 'rrf_score' in final_ranked_df.columns:
                 final_ranked_df = final_ranked_df.sort_values('rrf_score', ascending=False)
                 
            ranked_list = final_ranked_df['folder'].drop_duplicates().tolist()
            results[i]['RankedList'] = ranked_list

            i += 1
        return results
    
    def apply_document_level_rrf(self, dfs_dict):
        """
        Performs Reciprocal Rank Fusion on DOCUMENT lists.
        """
        processed_dfs = []
        for model in self.models:
            df = dfs_dict[model].copy()
            weight = RFF_WEIGHTS.get(model, 1.0)
            
            df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
            df['rank'] = df.index + 1
            df['rr_score'] = weight * (1 / (RRF_R_PARAMETER + df['rank']))
            
            processed_dfs.append(df[['docno', 'folder', 'rr_score']])

        combined_df = pd.concat(processed_dfs, ignore_index=True)
        final_df = combined_df.groupby(['docno', 'folder'], as_index=False)['rr_score'].sum()
        final_df = final_df.rename(columns={'rr_score': 'score'})
        return final_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    def apply_folder_level_rrf(self, dfs_dict):
        """
        Performs Reciprocal Rank Fusion on FOLDER lists.
        """
        processed_dfs = []
        for model in self.models:
            df = dfs_dict[model].copy()
            weight = RFF_WEIGHTS.get(model, 1.0)
            df.sort_values(by='score', ascending=False).reset_index(drop=True)
            df['rank'] = df.index + 1
            df['rr_score'] = weight * (1 / (RRF_R_PARAMETER + df['rank']))
            processed_dfs.append(df[['folder', 'rr_score']])

        combined_df = pd.concat(processed_dfs, ignore_index=True)
        final_df = combined_df.groupby('folder', as_index=False)['rr_score'].sum()
        final_df = final_df.rename(columns={'rr_score': 'rrf_score'})
        final_ranking = final_df.sort_values(by='rrf_score', ascending=False).reset_index(drop=True)
        return final_ranking[['folder', 'rrf_score']]
    
    def create_folder_relations_for_expansion(self, trainingSet):
        """
        Builds a graph of relationships between training documents and all known folders.
        
        This dictionary drives the expansion logic. It maps every target folder to lists of 'neighbor' documents from the training set based on heuristics:
        - Same Box
        - Same SNC (Classification Code)
        - Close Date + Same SNC
        - Similar SNC (Hierarchical match)
        """
        def close_enough(item, folder):
            """Checks if item date falls within folder's start/end dates."""
            itemDate = item['date']
            folderStart = folder['date']
            folderEnd = folder['endDate']
            if itemDate!='Unknown' and folderStart!='Unknown' and folderEnd!='Unknown':
                if datetime.strptime(folderStart, '%m/%d/%Y')<datetime.strptime(itemDate, '%Y-%m-%d') and datetime.strptime(folderEnd, '%m/%d/%Y')>datetime.strptime(itemDate, '%Y-%m-%d'):
                    return True
            return False

        def similar_snc(snc1, snc2):
            """Checks if two SNC codes share the same parent hierarchy."""
            s1=[snc1, '', '']
            if ' ' in snc1:
                s1[0], s1[1] = snc1.split()
                if '-' in s1[1]:
                    s1[1], s1[2] = s1[1].split('-')
            s2 = [snc2, '', '']
            if ' ' in snc2:
                s2[0], s2[1] = snc2.split()
                if '-' in s2[1]:
                    s2[1], s2[2] = s2[1].split('-')
            if s1[0]==s2[0]:
                if s1[0]!='POL':
                    return True
                elif s1[1]==s2[1]:
                    return True
            return False

        relations = {}
        for folder in self.folderMetadata:
            relations[folder] = {}
            relations[folder]['same folder'] = []
            relations[folder]['same box'] = []
            relations[folder]['adjacent box'] = []
            relations[folder]['same snc close date'] = []
            relations[folder]['same snc'] = []
            relations[folder]['similar snc'] = []

            for doc in trainingSet:
                if doc['folder']==folder: # Documents from the same folder as the current one
                    relations[folder]['same folder'].append(doc['docno'])
                if doc['box']==self.folderMetadata[folder]['box']: # Documents from the same box as the current one
                    relations[folder]['same box'].append(doc['docno'])
                if doc['box'][0]==self.folderMetadata[folder]['box'][0] and abs(int(doc['box'][1:])-int(self.folderMetadata[folder]['box'][1:]))<2: # Documents from the box after
                    #print(f"For docno {doc['docno']} Box {doc['box']} is adjacent to box {folderMetadata[folder]['box']}")
                    relations[folder]['adjacent box'].append(doc['docno'])
                file_snc = self.folderMetadata[doc['folder']]['snc']
                folder_snc = self.folderMetadata[folder]['snc']
                if file_snc!='Unknown' and file_snc==folder_snc: # Documents with the same SNC
                    relations[folder]['same snc'].append(doc['docno']) 
                    if close_enough(self.items[doc['docno']], self.folderMetadata[folder]):
                        relations[folder]['same snc close date'].append(doc['docno']) # Documents with same SNC and with date between the start and end of the folder
                if file_snc!='Unknown' and folder_snc!='Unknown' and similar_snc(file_snc,folder_snc): # Documents with Similar SNC in the first two levels
                    relations[folder]['similar snc'].append(doc['docno'])

        return relations

    def produce_expansion_results(self, result):
        """
        Applies the Expansion logic.
        
        1. Takes initial retrieval results (Direct Hits).
        2. Identifies 'empty' folders that were not retrieved.
        3. Uses `relations` to find neighbors of retrieved docs pointing to these empty folders.
        4. Calculates inferred scores based on neighbor evidence.
        5. Applies 'Safety Ceiling' to ensure inferred results don't outrank top direct hits.
        """
        scores = {}
        folder_score = {}
        new_folder_scores = {}
        
        for i, row in result.iterrows():
            scores[row['docno']] = row['score']
            if row['folder'] not in folder_score:
                folder_score[row['folder']] = row['score']
            elif row['score'] > folder_score[row['folder']]:
                folder_score[row['folder']] = row['score']

        for folder in self.relations:
            if folder not in folder_score: 
                candidate_sets = []
                technique_map = {
                    'same_box': 'same box',
                    'same_snc': 'same snc',
                    'close_date': 'same snc close date',
                    'similar_snc': 'similar snc'
                }

                for tech_arg in self.expansion:
                    rel_key = technique_map.get(tech_arg)
                    if rel_key:
                        docs = self.relations[folder][rel_key]
                        valid_docs = {doc for doc in docs if doc in scores}
                        candidate_sets.append(valid_docs)

                if not candidate_sets:
                    final_docs = set()
                else:
                    final_docs = candidate_sets[0]
                    for s in candidate_sets[1:]:
                        final_docs = final_docs.intersection(s)

                expansion_scores = [scores[doc] for doc in final_docs]

                if len(expansion_scores) > 0:
                    folder_score[folder] = statistics.mean(expansion_scores)
                    new_folder_scores[folder] = folder_score[folder]

        sorted_scores = sorted(scores.values(), reverse=True)

        top_score = sorted_scores[min(self.expansion_ceiling_k - 1, len(sorted_scores) - 1)]
        sorted_new_folder_scores = sorted(new_folder_scores.values(), reverse=True)

        if len(sorted_scores)>0 and len(sorted_new_folder_scores)>0 and sorted_new_folder_scores[0] > top_score:
            penalty = sorted_new_folder_scores[0] - top_score + 0.001
            for folder in new_folder_scores:
                folder_score[folder] = max(0, folder_score[folder] - penalty)
                new_folder_scores[folder] = folder_score[folder]

        folder_score = dict(sorted(folder_score.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(list(folder_score.items()), columns=['folder', 'score'])

        return df

    def saving_folder_name(self):
        """Generates a consistent, descriptive folder name for the experiment results."""
        search_field_name = ""
        if len(self.current_searching_field) >= 2:
            for field in self.current_searching_field:
                search_field_name += SEARCHING_FIELD_MAP[field]
        else:
            search_field_name = SEARCHING_FIELD_MAP[self.current_searching_field[0]]

        if self.all_folders_folder_label == True:
            search_field_name = "ALLFL"
        
        expansion_name = ""
        if len(self.expansion) > 0:
            for exp in self.expansion:
                expansion_name += EXPANSION_NAME_MAP[exp]+"-"
            expansion_name = expansion_name + f"{self.expansion_ceiling_k}-"
        else:
            expansion_name = "NEX-"
        query_fields_name = self.current_query_field
        model_name        = "-".join(self.models).upper()

        uneven = "-UNEVEN" if self.sampling == "uneven" else ""

        return f"4perBox-{search_field_name}{uneven}_{expansion_name[:-1]}_{query_fields_name}_{model_name}"

if __name__ == "__main__":
   gen = RunGenerator()
   gen.run_experiments()