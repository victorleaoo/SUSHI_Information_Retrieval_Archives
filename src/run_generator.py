import re
import os
import json
import random
import time
import sys
from datetime import datetime
import statistics
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import pyterrier as pt
from tqdm import tqdm
from pylate import indexes, models, retrieve

# Models Class
from models import BM25Model, EmbeddingsModel, ColBERTModel, BM_25_FIELD_WEIGHTS
from evaluator import Evaluator

class Style:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

# CONSTANTS
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FOLDER_METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'folders_metadata', 'FoldersV1.3.json')
ITEMS_METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'items_metadata', 'itemsV1.2.json')
SUSHI_FILES_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
TOPICS_PATH = os.path.join(PROJECT_ROOT, "src", "data_creation", "topics_output.txt")
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'RunResults.tsv')
FOLDER_QRELS_PATH = os.path.join(PROJECT_ROOT, 'qrels', 'formal-run-qrels', 'formal-folder-qrel.txt')
BOX_QRELS_PATH = os.path.join(PROJECT_ROOT, 'qrels', 'formal-run-qrels', 'formal-box-qrel.txt')
ALL_DOCUMENTS_ECF_PATH = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated', 'ECF_ALL_TRAINING_SET.json')

RANDOM_SEED_LIST = [1, 42, 100, 300, 333, 777, 999, 2025, 6159, 12345, 19865, 53819,
                    56782, 62537, 72738, 75259, 81236, 91823, 98665, 98765, 99009, 999777333,
                    120302, 123865, 170302, 180803, 5122025, 12052024, 12052025, 99000011]

SEARCHING_FIELD_MAP = {
    'folderlabel': 'F', 
    'ocr': 'O', 
    'summary': 'S', 
    'title': 'T'
}

EXPANSION_NAME_MAP = {
    'same_box': 'SB',
    'same_snc': 'SS',
    'similar_snc': 'SMS',
    'close_date': 'CD'
}

RFF_WEIGHTS = {
    'bm25': 1.0,
    'embeddings': 0.65,
    'colbert': 0.65
}
RRF_R_PARAMETER = 0  # "we set r = 0"

class RunGenerator:
    def __init__(self, 
                 searching_fields=[['folderlabel']],#, [['title', 'ocr']], [['title', 'summary']], [['ocr', 'summary']], [['title', 'ocr', 'summary']]], # [['title', 'ocr', 'folderlabel', 'summary']]
                 query_fields=['TD'], # ["T", "TD", "TDN"]
                 run_type='random', # 'random', 'all_documents'
                 models=['embeddings'], # list: 'bm25', 'embeddings', 'colbert'
                 sampling='uniform', # 'uniform', 'uneven' (only works for run_type == 'random')
                 expansion=[], # ['same_box', 'same_snc', 'similar_snc', 'close_date']
                 all_folders_folder_label=True
                 ):
        # Initialize params
        self.searching_fields = searching_fields
        self.query_fields = query_fields
        self.run_type = run_type
        self.models = models
        self.sampling = sampling
        self.expansion = expansion
        self.all_folders_folder_label = all_folders_folder_label

        # Load items and folderMetadata
        #print('> Loading Items and folderMetadata... <')
        with open(ITEMS_METADATA_PATH) as itemsMetadataFile:
            self.items = json.load(itemsMetadataFile)
        with open(FOLDER_METADATA_PATH) as folderMetadataFile:
            self.folderMetadata = json.load(folderMetadataFile)
        #print('!> Loaded Items and folderMetadata <!')

        self.evaluator = Evaluator(FOLDER_QRELS_PATH, BOX_QRELS_PATH)
        
        # Load full collection of data
        def sortLongest(my_dict):
            dict_lengths = {key: len(value) for key, value in my_dict.items()}
            sorted_keys = sorted(dict_lengths, key=lambda k: dict_lengths[k], reverse=True)
            sorted_dict = {key: my_dict[key] for key in sorted_keys}
            return sorted_dict
        
        # Load full test collection (necessary to generate ecf)
        self.fullCollection = {}
        for box in os.listdir(SUSHI_FILES_PATH):
            self.fullCollection[box] = {}
            for folder in os.listdir(os.path.join(SUSHI_FILES_PATH,box)):
                self.fullCollection[box][folder] = []
                for file in os.listdir(os.path.join(SUSHI_FILES_PATH,box,folder)):
                    self.fullCollection[box][folder].append(file)
        for box in self.fullCollection:
            self.fullCollection[box] = sortLongest(self.fullCollection[box])
        
    ##### MAIN WORKFLOW #####
    def run_experiments(self):
        for searching_field in self.searching_fields:
            for query_field in self.query_fields:
                self.current_searching_field = searching_field
                self.current_query_field = query_field
                print(f"{Style.BOLD}{Style.GREEN}> Running Experiments for:{Style.RESET}")
                print(f"\t- {Style.BOLD}Searching fields:{Style.RESET}  {Style.CYAN}{', '.join([field for field in self.current_searching_field] if isinstance(self.current_searching_field[0], str) else [field for field in self.current_searching_field[0]])}{Style.RESET}")
                print(f"\t- {Style.BOLD}Query fields:{Style.RESET}      {Style.CYAN}{self.current_query_field}{Style.RESET}")
                print(f"\t- {Style.BOLD}Run type:{Style.RESET}          {Style.CYAN}{self.run_type}{Style.RESET}")
                print(f"\t- {Style.BOLD}Model:{Style.RESET}             {Style.CYAN}{', '.join([model for model in self.models])}{Style.RESET}")
                print(f"\t- {Style.BOLD}Expansion Methods:{Style.RESET} {Style.CYAN}{', '.join([exp for exp in self.expansion])}{Style.RESET}")
                run_folder_name = self.saving_folder_name()
                metrics_output_folder = os.path.abspath(f'../all_runs/{run_folder_name}')
                if self.run_type == 'random':
                    #print('>> Running Random Experiment <<')
                    for random_seed in tqdm(RANDOM_SEED_LIST, desc="Generating Results of Random Runs"):
                        random.seed(random_seed)
                        self.random_seed = random_seed
                        # 1. Create the random ECF
                        self.ecf = self.create_random_ecf()

                        # 2. Get data dict
                        clean_data = self.prepare_training_data()

                        if self.run_type != "all_documents" and self.all_folders_folder_label == False:
                            self.relations = self.create_folder_relations_for_expansion(clean_data)

                        # 3. Train model
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
                        
                        # 4. Generate Search Results for Topics
                        results = self.produce_topics_results()

                        # 5. Save and Evaluate Results
                        run_name=f'45-Topics-Random-{random_seed}'
                        self.evaluator.save_run_file(results, RESULTS_PATH, run_name)
                        json_path = os.path.join(metrics_output_folder, f'Random{self.random_seed}_TopicsFolderMetrics.json')
                        self.evaluator.evaluate(RESULTS_PATH, json_path)
                    self.evaluator.generate_aggregated_metrics(metrics_output_folder, 'random')
                elif self.run_type == 'all_documents':
                    # 1. Read ECF
                    with open(ALL_DOCUMENTS_ECF_PATH) as ecfFile:
                        self.ecf = json.load(ecfFile)

                    # 2. Get data dict
                    clean_data = self.prepare_training_data()

                    # 3. Train model
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

                    # 4. Generate Search Results for Topics
                    results = self.produce_topics_results()

                    # 5. Save and Evaluate Results
                    self.evaluator.save_run_file(results, RESULTS_PATH, run_name)
                    json_path = os.path.join(metrics_output_folder, 'AllDocuments_TopicsFolderMetrics.json')
                    self.evaluator.evaluate(RESULTS_PATH, json_path)
                    self.evaluator.generate_aggregated_metrics(metrics_output_folder, 'all_documents')

    def create_random_ecf(self):
        # Load all topics
        with open(TOPICS_PATH, 'r', encoding='utf-8') as file:
            topics = list(json.load(file).values())

        if self.sampling == 'uniform':
            trainingSet = []
            trainingFiles = []
            max = 300
            docsPerBox = 5
            for box in self.fullCollection:
                
                folderDocs = [0]*max
                total = 0
                for j in range(docsPerBox): # 5 documents per box
                    # if the box has less than 5 folders, some folders will have more documents selected
                    for i in range(min(len(self.fullCollection[box]),docsPerBox,max)): # select the min between the len of folders in the box, 5 and 300
                        if total<docsPerBox:
                            folderDocs[i] += 1
                            total += 1

                folders_list = list(self.fullCollection[box].keys())
                random.shuffle(folders_list)

                i = 0
                for folder in folders_list:
                    valid_candidates = [doc for doc in self.fullCollection[box][folder] if doc not in trainingFiles]
                    num_to_pick = min(folderDocs[i], len(valid_candidates))
                    if num_to_pick > 0:
                        # random selection of valid documents candidates (not yet choosen)
                        selected = random.sample(valid_candidates, num_to_pick)
                        
                        for candidate in selected:
                            trainingFiles.append(candidate)
                            trainingSet.append(box+'/'+folder+'/'+candidate)
                    i+=1
                trainingSet.sort()

        ECF_PATH = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated', f'ECF_RANDOM_{self.random_seed}.json')
        ecf = {}
        ecf['ExperimentName'] = f'ECF w/ Random Seed {self.random_seed}'
        ecf['ExperimentSets'] = []
        ecf['ExperimentSets'].append({})
        ecf['ExperimentSets'][0]['TrainingDocuments'] = trainingSet
        ecf['ExperimentSets'][0]['Topics'] = {}
        for topic in range(len(topics)):
            topicId = topics[topic]['ID']
            ecf['ExperimentSets'][0]['Topics'][topicId] = topics[topic]
            topics[topic]['ID'] = topicId
        with open(ECF_PATH, 'w') as json_file:
            json.dump(ecf, json_file, indent=4)

        return ecf

    def prepare_training_data(self):
        trainingSet = []
        # Determine the fields for text_blob generation
        # Logic adapted from your original setup_embeddings
        current_fields = self.current_searching_field[0] if isinstance(self.current_searching_field[0], list) else [self.current_searching_field[0]]

        if not self.all_folders_folder_label:
            for trainingDoc in self.ecf["ExperimentSets"][0]["TrainingDocuments"]:
                file = trainingDoc[-10:-4] 
                folder = self.items[file]['Sushi Folder']
                box = self.items[file]['Sushi Box']
                docDate = self.items[file]['date']
                
                # 1. Base Metadata
                doc_entry = {
                    'docno': file,
                    'folder': folder,
                    'box': box,
                    'date': docDate,
                    'title': self.items[file]['title'],
                    'ocr': self.items[file]['ocr'][0],
                    'summary': self.items[file]['summary'],
                }

                # 2. Handle Folder Label
                try:
                    label = self.folderMetadata[folder]['label_parent_expanded'] + ". " + self.folderMetadata[folder]['scope_stoppers']
                except:
                    label = self.folderMetadata[folder]['label']
                doc_entry['folderlabel'] = label

                # 3. Generate 'text_blob' for Dense Retrievers (Concat logic)
                text_blob = ""
                for field in current_fields:
                    if field == 'folderlabel':
                        text_blob += label + ". "
                    else:
                        val = self.items[file][field]
                        if field == 'ocr': 
                            val = val[0] # Handle OCR list
                        text_blob += str(val) + ". "
                doc_entry['text_blob'] = text_blob.strip()

                trainingSet.append(doc_entry)
        else:
            for folder in self.folderMetadata:
                folder_entry = {
                    'docno': folder,
                    'folder': folder,
                    'date': self.folderMetadata[folder]['date'],
                    'box': self.folderMetadata[folder]['box']
                }
                try:
                    label = self.folderMetadata[folder]['label_parent_expanded'] + ". " + self.folderMetadata[folder]['scope_stoppers']
                except:
                    label = self.folderMetadata[folder]['label']
                folder_entry['folderlabel'] = label
                folder_entry['text_blob'] = label
                
                trainingSet.append(folder_entry)

        return trainingSet
    
    def produce_topics_results(self):
        #print(">>> Generating Results... <<<")
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

            dfs_to_combine = {}

            for model_name, model_instance in self.active_models.items():
                # 1. Get raw search results (DataFrame)
                raw_df = model_instance.search(query)
                
                # 2. Apply Expansion
                if len(self.expansion) == 0 or self.run_type == 'all_documents':
                    final_df = raw_df[['folder', 'score']]
                else:
                    final_df = self.produce_expansion_results(raw_df)
                
                dfs_to_combine[model_name] = final_df
            
            if len(self.models) == 1:
                # Get the only dataframe
                single_model = self.models[0]
                ranked_list = dfs_to_combine[single_model]['folder'].drop_duplicates().tolist()
            else:
                ranked_df = self.apply_weighted_rrf(dfs_to_combine)
                ranked_list = ranked_df['folder'].drop_duplicates().tolist()

            results[i]['RankedList'] = ranked_list

            i += 1
        return results
    
    def apply_weighted_rrf(self, dfs_dict):
        processed_dfs = []

        # dataframes ordered by score and with rank reciprocal rank
        for model in self.models:
            df = dfs_dict[model].copy()
            weight = RFF_WEIGHTS.get(model, 1.0)

            df.sort_values(by='score', ascending=False).reset_index(drop=True)
            df['rank'] = df.index + 1

            df['rr_score'] = weight * (1 / (RRF_R_PARAMETER + df['rank']))

            processed_dfs.append(df[['folder', 'rr_score']])

        # combine models results
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        final_df = combined_df.groupby('folder', as_index=False)['rr_score'].sum()
        final_df = final_df.rename(columns={'rr_score': 'rrf_score'})
        final_ranking = final_df.sort_values(by='rrf_score', ascending=False).reset_index(drop=True)
        
        return final_ranking[['folder', 'rrf_score']]
    
    def create_folder_relations_for_expansion(self, trainingSet):
        def close_enough(item, folder):
            itemDate = item['date']
            folderStart = folder['date']
            folderEnd = folder['endDate']
            if itemDate!='Unknown' and folderStart!='Unknown' and folderEnd!='Unknown':
                if datetime.strptime(folderStart, '%m/%d/%Y')<datetime.strptime(itemDate, '%Y-%m-%d') and datetime.strptime(folderEnd, '%m/%d/%Y')>datetime.strptime(itemDate, '%Y-%m-%d'):
                    #print(f"Found item {item['Sushi File']} in folder {item['Sushi Folder']} with date {itemDate} between folder start date {folderStart} and folder end date {folderEnd}")
                    return True
            return False

        def similar_snc(snc1, snc2):
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
        # Do the expansion
        scores = {}
        folder_score = {}
        new_folder_scores = {}
        for i, row in result.iterrows():
            scores[row['docno']] = row['score']
            if row['folder'] not in folder_score:
                folder_score[row['folder']]=row['score']
            elif row['score']>folder_score[row['folder']]:
                folder_score[row['folder']] = row['score']
        # Each folder receive the score of its highest scored document

        for folder in self.relations:
            if folder not in folder_score: # If the folder doesn't have a result (not in training docs probably)
                same_box=[]
                close_date=[]
                same_snc = []
                similar_snc = []
                for doc in self.relations[folder]['same box']: 
                    if doc in scores: # If document with score is from the same box, it saves 
                        same_box.append(scores[doc]) 
                for doc in self.relations[folder]['same snc close date']:
                    if doc in scores: # If document with score has the same snc and close date, it saves
                        close_date.append(scores[doc])
                for doc in self.relations[folder]['same snc']:
                    if doc in scores: # If document with score has the same snc, it saves
                        same_snc.append(scores[doc])
                for doc in self.relations[folder]['similar snc']:
                    if doc in scores: # If document with score has similar snc, it saves
                        similar_snc.append(scores[doc])
                
                # Uses only same box and close date
                expansion_set = []
                for expansion_technique in self.expansion:
                    #print(f"Using {expansion_technique} in Expansion") 
                    if expansion_technique == 'same_box':
                        expansion_set += same_box
                    if expansion_technique == 'close_date':
                        expansion_set += close_date
                    if expansion_technique == 'same_snc':
                        expansion_set += same_snc
                    if expansion_technique == 'similar_snc':
                        expansion_set += similar_snc
                        
                if len(expansion_set) > 0:
                    folder_score[folder] = statistics.mean(expansion_set) # The folder score will be the mean of the documents assigned during expansion process
                    new_folder_scores[folder] = folder_score[folder] # Scores for all folders, including those not in the training set
                    
        sorted_scores = sorted(scores.values(), reverse=True)

        max_rank = min(5,len(sorted_scores))
        sorted_new_folder_scores = sorted(new_folder_scores.values(), reverse=True)

        if max_rank>0 and len(sorted_new_folder_scores)>0 and sorted_new_folder_scores[0] > sorted_scores[max_rank-1]:
            for folder in new_folder_scores:
                folder_score[folder] = max(0,folder_score[folder]-sorted_new_folder_scores[0]+sorted_scores[max_rank-1]-0.001)
                new_folder_scores[folder] = folder_score[folder]

        sorted_new_folder_scores = sorted(new_folder_scores.values(), reverse=True)
        folder_score = dict(sorted(folder_score.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(list(folder_score.items()), columns=['folder', 'score'])

        return df

    def saving_folder_name(self):
        search_field_name = ""
        if isinstance(self.current_searching_field[0], list):
            for field in self.current_searching_field[0]:
                search_field_name += SEARCHING_FIELD_MAP[field]
        else:
            search_field_name = SEARCHING_FIELD_MAP[self.current_searching_field[0]]

        if self.all_folders_folder_label == True:
            search_field_name = "ALLFL"
        
        expansion_name = ""
        if len(self.expansion) > 0:
            for exp in self.expansion:
                expansion_name += EXPANSION_NAME_MAP[exp]+"-"
        else:
            expansion_name = "NEX"
        query_fields_name = self.current_query_field
        model_name        = "-".join(self.models).upper()

        return f"{search_field_name}_{expansion_name[:-1]}_{query_fields_name}_{model_name}"
    
if __name__ == "__main__":
    run_experiments = RunGenerator()
    run_experiments.run_experiments()