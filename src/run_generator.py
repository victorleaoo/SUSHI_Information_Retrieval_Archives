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
import pytrec_eval
import numpy as np
import scipy.stats as st
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from pylate import indexes, models, retrieve

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

BM_25_FIELD_WEIGHTS = {
    'title':       {'index_col': 'title',       'w': 2.3,  'c': 0.65},
    'ocr':         {'index_col': 'ocr',         'w': 0.1,  'c': 0.4},
    'folderlabel': {'index_col': 'folderlabel', 'w': 0.2,  'c': 0.6},
    'summary':     {'index_col': 'summary',     'w': 0.08, 'c': 2.0}
}

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
                 query_fields=['T', 'TD', 'TDN'], # ["T", "TD", "TDN"]
                 run_type='random', # 'random', 'all_documents'
                 models=['bm25'], # list: 'bm25', 'embeddings', 'colbert'
                 sampling='uniform', # 'uniform', 'uneven' (only works for run_type == 'random')
                 expansion=[] # ['same_box', 'same_snc', 'similar_snc', 'close_date']
                 ):
        # Initialize params
        self.searching_fields = searching_fields
        self.query_fields = query_fields
        self.run_type = run_type
        self.models = models
        self.sampling = sampling
        self.expansion = expansion

        # Initial configs if system is Windows
        if os.name == 'nt': # Windows
            self.unix_check = True
            java_home = r'C:\Program Files\Java\jdk-11'
            os.environ["JAVA_HOME"] = java_home
        else:
            self.unix_check = False

        # Load items and folderMetadata
        #print('> Loading Items and folderMetadata... <')
        with open(ITEMS_METADATA_PATH) as itemsMetadataFile:
            self.items = json.load(itemsMetadataFile)
        with open(FOLDER_METADATA_PATH) as folderMetadataFile:
            self.folderMetadata = json.load(folderMetadataFile)
        #print('!> Loaded Items and folderMetadata <!')
        
        # Load full collection of data
        def sortLongest(my_dict):
            dict_lengths = {key: len(value) for key, value in my_dict.items()}
            sorted_keys = sorted(dict_lengths, key=lambda k: dict_lengths[k], reverse=True)
            sorted_dict = {key: my_dict[key] for key in sorted_keys}
            return sorted_dict
        
        # Load full test collection (necessary to generate ecf)
        #print('> Loading full collection... <')
        self.fullCollection = {}
        for box in os.listdir(SUSHI_FILES_PATH):
            self.fullCollection[box] = {}
            for folder in os.listdir(os.path.join(SUSHI_FILES_PATH,box)):
                self.fullCollection[box][folder] = []
                for file in os.listdir(os.path.join(SUSHI_FILES_PATH,box,folder)):
                    self.fullCollection[box][folder].append(file)
        for box in self.fullCollection:
            self.fullCollection[box] = sortLongest(self.fullCollection[box])
        #print('!> Loaded full collection <!')

        self.MODELS_METHODS = {
            'bm25': (self.train_bm25, self.search_bm25),
            'embeddings': (self.setup_embeddings, self.search_embeddings),
            'colbert': (self.setup_colbert, self.search_colbert)
        }
        
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
                if self.run_type == 'random':
                    #print('>> Running Random Experiment <<')
                    for random_seed in tqdm(RANDOM_SEED_LIST, desc="Generating Results of Random Runs"):
                        random.seed(random_seed)
                        self.random_seed = random_seed
                        # 1. Create the random ECF
                        self.ecf = self.create_random_ecf()
                        #print(f"!>>> ECF created for seed {random_seed} <<<!")

                        # 2. Train model
                        for model in self.models:
                            self.MODELS_METHODS[model][0]()
                        
                        # 3. Generate Search Results for Topics
                        results = self.produce_topics_results()

                        # 4. Save and Evaluate Results
                        self.write_results(results)
                        self.evaluate_results()
                    self.metric_file_generator()
                elif self.run_type == 'all_documents':
                    # 1. Read ECF
                    with open(ALL_DOCUMENTS_ECF_PATH) as ecfFile:
                        self.ecf = json.load(ecfFile)

                    # 2. Train model
                    for model in self.models:
                        self.MODELS_METHODS[model][0]()

                    # 3. Generate Search Results for Topics
                    results = self.produce_topics_results()

                    # 4. Save and Evaluate Results
                    self.write_results(results)
                    self.evaluate_results()
                    self.metric_file_generator()

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
                query = f"{title} {description}".strip()
            else:
                query = title.strip()

            if len(self.models) == 1:
                if self.models[0] == 'embeddings':
                    rankedFolderList = self.search_embeddings(query)
                elif self.models[0] == "bm25":
                    rankedFolderList = self.search_bm25(query)
                elif self.models[0] == "colbert":
                    rankedFolderList = self.search_colbert(query)

                results[i]['RankedList'] = rankedFolderList['folder'].drop_duplicates().tolist()
            else: # rrf = True -> the result is going to be a combination of results
                dfs_to_combine = {}
                for model in self.models:
                    dfs_to_combine[model] = self.MODELS_METHODS[model][1](query)
                    
                rankedFolderList = self.apply_weighted_rrf(dfs_to_combine)
                results[i]['RankedList'] = rankedFolderList['folder'].drop_duplicates().tolist()
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
    
    def train_bm25(self):
        unix = self.unix_check

        trainingSet=[]

        # Creating trainingSet list with documents metadata
        for trainingDoc in self.ecf["ExperimentSets"][0]["TrainingDocuments"]:
            file = trainingDoc[-10:-4] # This extracts the file name and ignores the box and folder labels, which we will get from the metadata
            folder = self.items[file]['Sushi Folder']
            box = self.items[file]['Sushi Box']
            try:
                label = self.folderMetadata[folder]['label_parent_expanded'] + self.folderMetadata[folder]['scope_stoppers']
            except:
                label = self.folderMetadata[folder]['label']
            docDate = self.items[file]['date']
            title = self.items[file]['title']
            ocr = self.items[file]['ocr'][0] # This indexes only the first page of the OCR
            summary = self.items[file]['summary'] # This comes from gpt4o-mini
            trainingSet.append({'docno': file, 
                                'folder': folder, 
                                'box': box,
                                'title': title, 
                                'ocr': ocr, 
                                'folderlabel': label, 
                                'summary': summary, 
                                'date': docDate})
        
        if self.run_type != "all_documents":
            self.relations = self.create_folder_relations_for_expansion(trainingSet)

        if not pt.java.started():
            pt.java.init()
        pt.ApplicationSetup.setProperty("terrier.use.memory.mapping", "false")
        pt.java.set_log_level('ERROR')

        # Setting BM25F weights
        if isinstance(self.current_searching_field[0], list):
            active_text_attrs = []
            bm25f_controls = {}

            field_idx = 0
            for field in self.current_searching_field[0]:
                if field in BM_25_FIELD_WEIGHTS:
                    col_name = BM_25_FIELD_WEIGHTS[field]['index_col']
                    active_text_attrs.append(col_name)
                    
                    weight = BM_25_FIELD_WEIGHTS[field]['w']
                    c_val = BM_25_FIELD_WEIGHTS[field]['c']
                    
                    bm25f_controls[f'w.{field_idx}'] = weight
                    bm25f_controls[f'c.{field_idx}'] = c_val
                    
                    field_idx += 1
                else:
                    print(f"{field} not available")
                    sys.exit()
            final_wmodel = "BM25F"
            final_controls = bm25f_controls 
        else:
            final_wmodel = "BM25"
            final_controls = {}
            active_text_attrs = [field for field in self.current_searching_field if field in BM_25_FIELD_WEIGHTS]

        #print(final_wmodel, final_controls, active_text_attrs)

        # Creating TerrierIndex with Unique ID as directory (need to work on Windows)
        indexDir = os.path.join(os.path.abspath("terrierindex"), str(int(time.time())))
        indexer = pt.IterDictIndexer(indexDir, 
                                    meta={'docno': 20, 'folder': 20, 'box': 20, 'date': 10}, 
                                    text_attrs=active_text_attrs,
                                    meta_reverse=['docno'], overwrite=True, fields=True
                                    )
        indexref = indexer.index(trainingSet)
        index = pt.IndexFactory.of(indexref)

        # Setting up model (BM25 or BM25F, depending on the config)
        self.BM25 = pt.terrier.Retriever(index, 
                                         wmodel=final_wmodel,
                                         controls=final_controls, 
                                         metadata=['docno', 'folder', 'box', 'date'], 
                                         num_results=1000
                                         )

        return
    
    def search_bm25(self, query):
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query) # Terrier fails if punctuation is found in a query
        result = self.BM25.search(query)
        #    qid  docid   docno     folder    box        date           rank      score        query
        #    1     47     S25507    B99990565 B0003      1966-04-27     0         8.366143     Future space missions
        #    1    138     S24275    E99990997 E0026      1969-05-01     1         8.183247     Future space missions
        rankedList = result[['folder', 'score']]

        if len(self.expansion) == 0 or self.run_type == 'all_documents':
            return rankedList
        
        return self.produce_expansion_results(result)
    
    def setup_embeddings(self):
        # 'all-MiniLM-L6-v2' is fast and effective. Use 'all-mpnet-base-v2' for higher accuracy (slower)
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        # Creating ids and text embeddings
        self.documents_ids = [doc[16:-4] for doc in self.ecf['ExperimentSets'][0]['TrainingDocuments']]

        self.documents_texts = []
        for doc in self.documents_ids:
            current_text = ""
            if isinstance(self.current_searching_field[0], str):
                if self.current_searching_field[0] == 'folderlabel':
                    try:
                        current_text = self.folderMetadata[folder]['label_parent_expanded'] + self.folderMetadata[folder]['scope_stoppers']
                    except:
                        current_text = self.folderMetadata[folder]['label']
                else:
                    current_text = self.items[doc][self.current_searching_field[0]] if self.current_searching_field[0] != 'ocr' else self.items[doc][self.current_searching_field[0]][0]
            else:
                for field in self.current_searching_field[0]:
                    if field == 'folderlabel':
                        current_text += self.folderMetadata[self.items[doc]['Sushi Folder']]['label_parent_expanded'] + self.folderMetadata[self.items[doc]['Sushi Folder']]['scope_stoppers'] + ". "
                    else:
                        text_to_add = self.items[doc][field]+". " if field != 'ocr' else self.items[doc][field][0]+". "
                        current_text += text_to_add
            self.documents_texts.append(current_text)

        trainingSet = []

        for trainingDoc in self.ecf["ExperimentSets"][0]["TrainingDocuments"]:
            file = trainingDoc[-10:-4] # This extracts the file name and ignores the box and folder labels, which we will get from the metadata
            folder = self.items[file]['Sushi Folder']
            box = self.items[file]['Sushi Box']
            trainingSet.append({'docno': file, 
                                'folder': folder, 
                                'box': box,
                                })

        if self.run_type != "all_documents":
            self.relations = self.create_folder_relations_for_expansion(trainingSet)

        self.documents_texts_embeddings = self.embedding_model.encode(self.documents_texts, convert_to_tensor=True)
        return
    
    def search_embeddings(self, query):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        cosine_scores = util.cos_sim(query_embedding, self.documents_texts_embeddings)[0]
        scores_and_ids = zip(cosine_scores.tolist(), self.documents_ids)

        ranked_ids = sorted(scores_and_ids, key=lambda x: x[0], reverse=True)

        rankedFolders = []
        results_data = []
        for score, id in ranked_ids:
            rankedFolders.append(self.items[id]['Sushi Folder'])

            results_data.append({
                'docno': id,
                'folder': self.items[id]['Sushi Folder'],
                'score': score
            })

        df = pd.DataFrame(results_data)

        if len(self.expansion) == 0 or self.run_type == 'all_documents':
            return df

        return self.produce_expansion_results(df)
    
    def setup_colbert(self):
        self.colbert_model = models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0")
        self.colbert_index = indexes.PLAID(
            index_folder="pylate-index",
            index_name="index",
            override=True,
        )
        self.colbert_retriever = retrieve.ColBERT(index=self.colbert_index)

        self.documents_ids = [doc[16:-4] for doc in self.ecf['ExperimentSets'][0]['TrainingDocuments']]
        for doc in self.documents_ids:
            current_text = ""
            if isinstance(self.current_searching_field[0], str):
                if self.current_searching_field[0] == 'folderlabel':
                    try:
                        current_text = self.folderMetadata[folder]['label_parent_expanded'] + self.folderMetadata[folder]['scope_stoppers']
                    except:
                        current_text = self.folderMetadata[folder]['label']
                else:
                    current_text = self.items[doc][self.current_searching_field[0]] if self.current_searching_field[0] != 'ocr' else self.items[doc][self.current_searching_field[0]][0]
            else:
                for field in self.current_searching_field[0]:
                    if field == 'folderlabel':
                        current_text += self.folderMetadata[self.items[doc]['Sushi Folder']]['label_parent_expanded'] + self.folderMetadata[self.items[doc]['Sushi Folder']]['scope_stoppers'] + ". "
                    else:
                        text_to_add = self.items[doc][field]+". " if field != 'ocr' else self.items[doc][field][0]+". "
                        current_text += text_to_add
            self.documents_texts.append(current_text)

        documents_embeddings = self.colbert_model.encode(
            self.documents_texts,
            batch_size=128,
            is_query=False, # Encoding documents
            show_progress_bar=False,
        )

        self.colbert_index.add_documents(
            documents_ids=self.documents_ids,
            documents_embeddings=documents_embeddings,
        )

        trainingSet = []

        for trainingDoc in self.ecf["ExperimentSets"][0]["TrainingDocuments"]:
            file = trainingDoc[-10:-4] # This extracts the file name and ignores the box and folder labels, which we will get from the metadata
            folder = self.items[file]['Sushi Folder']
            box = self.items[file]['Sushi Box']
            trainingSet.append({'docno': file,
                                'folder': folder, 
                                'box': box,
                                })

        if self.run_type != "all_documents":
            self.relations = self.create_folder_relations_for_expansion(trainingSet)
        
        return
    
    def search_colbert(self, query):
        query_embeddings = self.colbert_model.encode(
            [query],
            batch_size=128,
            is_query=True, # Encoding queries
            show_progress_bar=False,
        )

        results = self.colbert_retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=100,
        )

        df = pd.DataFrame(results[0])

        df['folder'] = df['id'].apply(lambda x: self.items[x]['Sushi Folder'])
        df['docno'] = df['id']

        if len(self.expansion) == 0 or self.run_type == 'all_documents':
            return df

        return self.produce_expansion_results(df)
    
    def write_results(self, results):
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)  # Ensure output directory exists
        name_run = f'45-Topics-Random-{self.random_seed}' if self.run_type == 'random' else '45-Topics-AllDocuments'
        with open(RESULTS_PATH, 'w') as f:
            for topic in results:
                for i in range(len(topic['RankedList'])):
                    print(f'{topic["Id"]}\t{topic["RankedList"][i]}\t{i+1}\t{1/(i+1):.4f}\t{name_run}', file=f)
        f.close()

    def saving_folder_name(self):
        search_field_name = ""
        if isinstance(self.current_searching_field[0], list):
            for field in self.current_searching_field[0]:
                search_field_name += SEARCHING_FIELD_MAP[field]
        else:
            search_field_name = SEARCHING_FIELD_MAP[self.current_searching_field[0]]
        
        expansion_name = ""
        if len(self.expansion) > 0:
            for exp in self.expansion:
                expansion_name += EXPANSION_NAME_MAP[exp]+"-"
        else:
            expansion_name = "NEX"
        query_fields_name = self.current_query_field
        model_name        = "-".join(self.models).upper()

        return f"{search_field_name}_{expansion_name[:-1]}_{query_fields_name}_{model_name}"

    def evaluate_results(self):
        measures = {'ndcg_cut', 'map', 'recip_rank', 'success'} # Generic measures for configuring a pytrec_eval evaluator
        
        with open(RESULTS_PATH) as runFile, open(FOLDER_QRELS_PATH) as folderQrelsFile, open(BOX_QRELS_PATH) as boxQrelsFile:
            folderRun = {}

            for line in runFile:
                topicId, folderId, _, score, _ = line.split('\t')
                if topicId not in folderRun:
                    folderRun[topicId] = {}
                folderRun[topicId][folderId] = float(score)

            folderQrels = {}
            for line in folderQrelsFile:
                topicId, _, folderId, relevanceLevel = line.split('\t')
                if topicId not in folderQrels:
                    folderQrels[topicId] = {}
                folderQrels[topicId][folderId] = int(relevanceLevel.strip())  # this deletes the \n at end of line
            folderEvaluator = pytrec_eval.RelevanceEvaluator(folderQrels, measures)
            folderTopicResults = folderEvaluator.evaluate(folderRun)  # replace run with folderQrels to see perfect evaluation measures

            file_path = f'../all_runs/{self.saving_folder_name()}/Random{self.random_seed}_TopicsFolderMetrics.json' if self.run_type == 'random' else f'../all_runs/{self.saving_folder_name()}/AllDocuments_TopicsFolderMetrics.json'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(folderTopicResults, f, indent=4)
        
        return 

    def calculate_mean_margin(self, values):
        n = len(values)

        mean = np.mean(values)

        if mean == 0.0:
            return 0.0, 0.0

        # (Standard Error of Mean - SEM)
        # s / sqrt(n)
        sem = st.sem(values) 
        
        # T-Student
        interval = st.t.interval(0.95, df=n-1, loc=mean, scale=sem)
        
        margin_of_error = interval[1] - mean
        
        return mean, margin_of_error

    def metric_file_generator(self):
        run_results_path = f"../all_runs/{self.saving_folder_name()}"
        if self.run_type == 'random':
            valid_files = [f for f in os.listdir(run_results_path) if f.startswith("Random") and f.endswith(".json")]
            topic_accumulator = {topic: [] for topic in {f"T{i}" for i in range(1, 46)}}

            for filename in valid_files:
                with open(os.path.join(run_results_path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                run_values = {}
                for raw_key, metrics in data.items():
                    norm_key = int(re.search(r'\d+$', raw_key).group())
                    val = metrics.get('ndcg_cut_5', 0.0)
                    run_values[norm_key] = val
                
                # Add to accumulator (fill missing topics with 0.0 for this run)
                for topic in {f"T{i}" for i in range(1, 46)}:
                    topic_accumulator[topic].append(run_values.get(int(topic[1:]), 0.0))

            with open(os.path.join(os.path.join(run_results_path, "topics_values.json")), 'w') as f:
                json.dump(topic_accumulator, f, indent=4)

            topics_intervals = {}
            for topic, values in topic_accumulator.items():
                mean, margin = self.calculate_mean_margin(values)
                lower = max(0.0, mean - margin)
                topics_intervals[topic] = (lower, mean, mean + margin)
            
            with open(os.path.join(run_results_path, "topics_mean_margin.json"), 'w') as f:
                json.dump(topics_intervals, f, indent=4)

            all_topics_means = [val[1] for val in topics_intervals.values()]
            global_mean, global_margin = self.calculate_mean_margin(all_topics_means)
            
            model_stats = {
                "model_global_ndcg": {
                    "mean": global_mean,
                    "margin": global_margin,
                    "interval": [(max(0.0, (global_mean - global_margin))), global_mean, (min(1.0, (global_mean + global_margin)))]
                }
            }

            print(model_stats)

            with open(os.path.join(run_results_path, "model_overall_stats.json"), 'w') as f:
                json.dump(model_stats, f, indent=4)
            
        elif self.run_type == 'all_documents':
            expected_topics = {f"T{i}" for i in range(1, 46)}
            with open(os.path.join(run_results_path, "AllDocuments_TopicsFolderMetrics.json"), 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
                data = {}
                for k, v in raw_data.items():
                    data[int(re.search(r'\d+$', k).group())] = v

            topics_intervals = {}

            for topic in expected_topics:
                if int(topic[1:]) in data:
                    val = data[int(topic[1:])]['ndcg_cut_5']
                    topics_intervals[topic] = (val, val, val)
                else:
                    topics_intervals[topic] = (0.0, 0.0, 0.0)
            
            all_topics_means = [val[1] for val in topics_intervals.values()]
            global_mean, global_margin = self.calculate_mean_margin(all_topics_means)
            
            model_stats = {
                "model_global_ndcg": {
                    "mean": global_mean,
                    "margin": global_margin,
                    "interval": [(max(0.0, (global_mean - global_margin))), global_mean, (min(1.0, (global_mean + global_margin)))]
                }
            }

            print(model_stats)

            with open(os.path.join(run_results_path, "all_documents_model_overall_stats.json"), 'w') as f:
                json.dump(model_stats, f, indent=4)
    
if __name__ == "__main__":
    run_experiments = RunGenerator()
    run_experiments.run_experiments()