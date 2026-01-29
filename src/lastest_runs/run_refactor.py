import os
import sys
import json
from datetime import datetime, date
import shutil
import re
import math
import pandas as pd

import statistics

import pyterrier as pt
import pytrec_eval

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Setting base path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_creation.SNCLabelTranslate import load_snc_expansions

def load_folder_items_metadata(folder_metadata_path, items_metadata_path):
    # items metadata
    with open(items_metadata_path) as itemsMetadataFile:
        items = json.load(itemsMetadataFile)
    
    print(f"-> {len(items)} items")

    # folder metadata
    with open(folder_metadata_path) as folderMetadataFile:
        folderMetadata = json.load(folderMetadataFile)

    # 

    # count = 0
    # count_total = len(folderMetadata)

    # for folder, _ in folderMetadata.items():
    #     snc = folderMetadata[folder]['snc']
    #     naraLabel = str(df_items.loc[df_items['Sushi Folder'] == folder, 'NARA Folder Name'].iloc[0])
    #     brownLabel = str(df_items.loc[df_items['Sushi Folder'] == folder, 'Brown Folder Name'].iloc[0])

    #     sncTranslation = load_snc_expansions('../src/SncTranslationV1.3.xlsx')

    #     if snc in sncTranslation:
    #         label = sncTranslation[snc]['expanded'] + ' ' + sncTranslation[snc]['scope']
    #         print(f"== SNC: {label}")
    #     elif naraLabel != 'nan':
    #         label = naraLabel
    #         print(f"== NARALABEL: {label}")
    #     elif brownLabel != 'nan':
    #         label = brownLabel
    #         print(f"== BROWNLABEL: {label}")
    #     else:
    #         label = 'Unknown'
    #         print("NO LABEL")

    #     folderMetadata[folder]['folder_label'] = label
    #     print(f'{count}/{count_total}')
    #     count += 1
    
    # with open('../data/folders_metadata/FoldersV1.2.json', 'w', encoding='utf-8') as f:
    #     json.dump(folderMetadata, f, indent=4)

    print(f"-> {len(folderMetadata)} folders")

    return folderMetadata, items

def readExperimentControlFile(fileName):
    with open(fileName) as ecfFile:
        ecf = json.load(ecfFile)
    return ecf

def NEWtrainTerrierModel(trainingDocs, searchFields, items, folderMetadata):
    global seq  # Used to control creation of a separate index for each training set

    if os.name == 'nt': # Windows
        unix_check = True
        java_home = r'C:\Program Files\Java\jdk-11'
        os.environ["JAVA_HOME"] = java_home
    else:
        unix_check = False

    unix = unix_check

    noShortOcr = False # Set to true if you want to replace OCR text that is nearly empty with the document title
    expanding = False  # Set to True if you want to index more than the full collection and not just the folders in the original training set

    print("- Training New Terrier Model")
    # print("\t**Params of the model:")
    # print(f"\t\tReplace Short Ocr for Title?: {noShortOcr}")
    # print(f"\t\tExpanding to index more than the folders in the training set?: {expanding}\n")

    #g = open('./files/titles.txt', 'w')

    trainingSet=[]

    print("\t**Building data structure indexed by Terrier...")
    # Build the data structure that Terrier will index (list of dicts, one per indexed item)
    for trainingDoc in trainingDocs:
        file = trainingDoc[-10:-4] # This extracts the file name and ignores the box and folder labels, which we will get from the metadata
        folder = items[file]['Sushi Folder']
        box = items[file]['Sushi Box']
        label = folderMetadata[folder]['folder_label']
        docDate = items[file]['date']
        title = items[file]['title']
        ocr = items[file]['ocr'][0] # This indexes only the first page of the OCR
        summary = items[file]['summary'] # This comes from gpt4o-mini

        # Optionally replace any hopelessly short OCR with the document title
        if noShortOcr and len(ocr)<5:
            print(f'Replaced OCR: //{ocr}// with Title //{title}//')
            ocr = title

        dateString = docDate
        #if docDate:
        #    dateString=docDate.strftime("%Y-%m-%d")
        #else:
        #    dateString = 'Unknown'

        #if len(label)==0 or len(file)==0 or len(folder)==0 or len(box)==0 or len(title)==0 or len(ocr)==0:
        #    print(f'folder {folder} docno {file} box {box} title {len(title)} ocr {len(ocr)}')
        trainingSet.append({'docno': file, 'folder': folder, 'box': box, 'title': title, 'ocr': ocr, 'folderlabel': label, 'summary': summary, 'date': dateString})
    print(f"\t\t{len(trainingSet)} documents with terrier index structure")
    print("\t**Data structure for Terrier built.\n")

    print("\t**Building relations for folders...")
    relations = find_relations(trainingSet, items, folderMetadata)
    print(f"\t\t{len(relations)} folders with relations created")
    print("\t**Relations for folders built.\n")

    import time
    # Create the Terrier index for this training set and then return a Terrier retriever for that index
    seq += 1 # We create one Terrier index per training set
    unique_id = str(int(time.time()))
    base_index_folder = os.path.abspath("terrierindex")
    indexDir = os.path.join(base_index_folder, unique_id)
    #if not pt.java.started(): pt.init()

    print("\t**Setting Up BM25F model...")
    #w = [2.3,0.1,0.2,0.08]
    #c = [0.65,0.4,0.6,2]

    field_config = {
        'title':       {'index_col': 'title',       'w': 2.3,  'c': 0.65},
        'ocr':         {'index_col': 'ocr',         'w': 0.1,  'c': 0.4},
        'folderlabel': {'index_col': 'folderlabel', 'w': 0.2,  'c': 0.6},
        'summary':     {'index_col': 'summary',     'w': 0.08, 'c': 2.0}
    }

    active_text_attrs = []
    bm25f_controls = {}

    field_idx = 0
    for field in searchFields:
        if field in field_config:
            col_name = field_config[field]['index_col']
            active_text_attrs.append(col_name)
            
            weight = field_config[field]['w']
            c_val = field_config[field]['c']
            
            bm25f_controls[f'w.{field_idx}'] = weight
            bm25f_controls[f'c.{field_idx}'] = c_val
            
            field_idx += 1
        else:
            print(f"{field} not available")
            sys.exit()

    if not pt.started():
        pt.java.init()

    pt.ApplicationSetup.setProperty("terrier.use.memory.mapping", "false")

    indexer = pt.IterDictIndexer(indexDir, 
                                 meta={'docno': 20, 'folder': 20, 'box': 20, 'date': 10}, 
                                 text_attrs=active_text_attrs,
                                 meta_reverse=['docno'], overwrite=True, fields=True
                                )
    indexref = indexer.index(trainingSet)
    index = pt.IndexFactory.of(indexref)

    if len(active_text_attrs) > 1:
        print(f"\t\t-> Multi-field index detected ({len(active_text_attrs)} fields). Using BM25F.")
        final_wmodel = "BM25F"
        final_controls = bm25f_controls
    else:
        print(f"\t\t-> Single-field index detected. Switching to standard BM25 (ignoring BM25F weights).")
        final_wmodel = "BM25"
        final_controls = {}

    BM25 = pt.terrier.Retriever(index, 
                                wmodel=final_wmodel,
                                controls=final_controls, 
                                metadata=['docno', 'folder', 'box', 'date'], 
                                num_results=1000
                               )
    print("\t**BM25F model set up.\n")
    return BM25, relations

def find_relations(training_set, items, folderMetadata):
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
    for folder in folderMetadata:
        relations[folder] = {}
        relations[folder]['same folder'] = []
        relations[folder]['same box'] = []
        relations[folder]['adjacent box'] = []
        relations[folder]['same snc close date'] = []
        relations[folder]['same snc'] = []
        relations[folder]['similar snc'] = []

        for doc in training_set:
            if doc['folder']==folder: # Documents from the same folder as the current one
                relations[folder]['same folder'].append(doc['docno'])
            if doc['box']==folderMetadata[folder]['box']: # Documents from the same box as the current one
                relations[folder]['same box'].append(doc['docno'])
            if doc['box'][0]==folderMetadata[folder]['box'][0] and abs(int(doc['box'][1:])-int(folderMetadata[folder]['box'][1:]))<2: # Documents from the box after
                #print(f"For docno {doc['docno']} Box {doc['box']} is adjacent to box {folderMetadata[folder]['box']}")
                relations[folder]['adjacent box'].append(doc['docno'])
            file_snc = folderMetadata[doc['folder']]['snc']
            folder_snc = folderMetadata[folder]['snc']
            if file_snc!='Unknown' and file_snc==folder_snc: # Documents with the same SNC
                relations[folder]['same snc'].append(doc['docno']) 
                if close_enough(items[doc['docno']], folderMetadata[folder]):
                    relations[folder]['same snc close date'].append(doc['docno']) # Documents with same SNC and with date between the start and end of the folder
            if file_snc!='Unknown' and folder_snc!='Unknown' and similar_snc(file_snc,folder_snc): # Documents with Similar SNC in the first two levels
                relations[folder]['similar snc'].append(doc['docno'])
    return relations

def trainModel(trainingDocuments, searchFields, items, folderMetadata):
    global seq
    global model
    print(f'\n=> Training Called, preparing index for experiment set {seq+1}')
    return NEWtrainTerrierModel(trainingDocuments, searchFields, items, folderMetadata)

def terrierSearch(query, engine, relations, expansion=True):
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query) # Terrier fails if punctuation is found in a query
    result = engine.search(query)
    #    qid  docid   docno     folder    box        date           rank      score        query
    #    1     47     S25507    B99990565 B0003      1966-04-27     0         8.366143     Future space missions
    #    1    138     S24275    E99990997 E0026      1969-05-01     1         8.183247     Future space missions
    rankedList = result['folder']

    if not expansion:
        rankedList = rankedList.drop_duplicates()
        return rankedList.tolist()

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

    for folder in relations:
        if folder not in folder_score: # If the folder doesn't have a result (not in training docs probably)
            same_box=[]
            close_date=[]
            same_snc = []
            similar_snc = []
            for doc in relations[folder]['same box']: 
                if doc in scores: # If document with score is from the same box, it saves 
                    same_box.append(scores[doc]) 
            for doc in relations[folder]['same snc close date']:
                if doc in scores: # If document with score has the same snc and close date, it saves
                    close_date.append(scores[doc])
            for doc in relations[folder]['same snc']:
                if doc in scores: # If document with score has the same snc, it saves
                    same_snc.append(scores[doc])
            for doc in relations[folder]['similar snc']:
                if doc in scores: # If document with score has similar snc, it saves
                    similar_snc.append(scores[doc])
            
            # Uses only same box and close date
            expansion_set = same_box+same_snc # or same_snc or similar_snc
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
    expanded_list = list(folder_score.keys())

    rankedList.drop_duplicates(inplace=True)

    dates = result['date']
    #write_dates(query, dates.tolist())
    #plot_dates(query, dates, pdf, 10)

    return expanded_list

def write_dates(query, dates):
    f = open('./dates/'+query+'.txt', 'w')
    print(f'\nQuery: {query}', file=f)
    for i in range(len(dates)):
        print(dates[i], file=f)
    f.close()

def generateSearchResults(ecf, searchFields, items, folderMetadata, queryFields="T", expansion=True):
    results = []
    i = 0
    global seq
    seq = 0
    for experimentSet in ecf['ExperimentSets']:
        index, relations = trainModel(experimentSet['TrainingDocuments'], searchFields, items, folderMetadata)

        #pdf = PdfPages('./dates/plots'+str(i+1)+'.pdf')

        topics = list(experimentSet['Topics'].keys())
        print("\t**Performing topics ranking with terrierSearch BM25F...")

        for j in range(len(topics)):
            results.append({})
            results[i]['Id'] = topics[j]

            title = experimentSet['Topics'][topics[j]].get('TITLE', '')
            description = experimentSet['Topics'][topics[j]].get('DESCRIPTION', '')
            narrative = experimentSet['Topics'][topics[j]].get('NARRATIVE', '')

            if queryFields == "TDN":
                query = f"{title} {description} {narrative}"
            elif queryFields == "TD":
                query = f"{title} {description}"
            else:
                query = title
            
            query = query.strip()

            rankedFolderList = terrierSearch(query, index, relations, expansion)
            results[i]['RankedList'] = rankedFolderList
            i+=1
        print("\t**Topics Ranking ended.")
        #pdf.close()
    return results

def writeSearchResults(fileName, results, runName):
    os.makedirs(os.path.dirname(fileName), exist_ok=True)  # Ensure output directory exists
    with open(fileName, 'w') as f:
        for topic in results:
            for i in range(len(topic['RankedList'])):
                print(f'{topic["Id"]}\t{topic["RankedList"][i]}\t{i+1}\t{1/(i+1):.4f}\t{runName}', file=f)
    f.close()

def createFolderToBoxMap():
    boxMap = {}
    with open(os.path.join(PROJECT_ROOT, 'data', 'folders_metadata', 'FoldersV1.2.json')) as foldersFile:
        folders = json.load(foldersFile)

    for folder in folders.items():
        boxMap[folder[0]] = folder[1]['box']
    
    return boxMap

def makeBoxRun(folderRun):
    global prefix
    boxMap = createFolderToBoxMap()
    boxRun = {}
    for topicId in folderRun:
        boxRun[topicId] = {}
        for folder in folderRun[topicId]:
            if boxMap[folder] not in boxRun[topicId]:
               boxRun[topicId][boxMap[folder]] = folderRun[topicId][folder]
    return boxRun

def stats(results, measure):
    sum=0
    squaredev=0
    n = len(results)
    for topic in results:
        sum += results[topic][measure]
    mean = sum / n
    for topic in results:
        squaredev += (results[topic][measure]-mean)**2
    variance = squaredev / (n-1)
    conf = 1.96 * math.sqrt(variance) / math.sqrt(n)
    return mean, conf

def process_topics_enrichment_strict(metrics_data, ecf_data, qrels_file, folder_run_data):
    # { "T18Eval-00001": {"FolderA", "FolderB", ...} }
    topic_to_training_folders = {}

    for exp_set in ecf_data.get('ExperimentSets', []):
        training_folders = set()
        raw_docs = exp_set.get('TrainingDocuments', [])
        
        # Extrair Folder ID do path: "Box/FolderID/DocID.pdf"
        for doc_path in raw_docs:
            parts = doc_path.split('/')
            if len(parts) >= 2:
                folder_id = parts[1]
                training_folders.add(folder_id)
        
        # O set de treino vale para todos os tópicos deste ExperimentSet
        topics_in_set = exp_set.get('Topics', {}).keys()
        for topic_id in topics_in_set:
            topic_to_training_folders[topic_id] = training_folders
    
    # { "topic_id": { "strict_relevant": {set}, "strict_highly": {set} } }
    qrels_map = {}

    with open(qrels_file, 'r', encoding='utf-8') as qrels_file:
        for line in qrels_file:
            parts = line.strip().split()
            if len(parts) < 4: continue
            
            topic_id = parts[0]
            folder_id = parts[2]
            try:
                relevance = int(parts[3])
            except ValueError:
                continue

            if topic_id not in qrels_map:
                qrels_map[topic_id] = {'strict_relevant': set(), 'strict_highly': set()}

            if relevance == 1:
                qrels_map[topic_id]['strict_relevant'].add(folder_id)
            elif relevance == 3:
                qrels_map[topic_id]['strict_highly'].add(folder_id)
    
    for topic_id, _ in metrics_data.items():
        qrels_sets = qrels_map.get(topic_id, {'strict_relevant': set(), 'strict_highly': set()})
        
        folders_score_1 = qrels_sets['strict_relevant']
        folders_score_3 = qrels_sets['strict_highly']

        folders_training = topic_to_training_folders.get(topic_id, set())

        metrics_data[topic_id]["count_relevant_folders_training"] = len(folders_score_1.intersection(folders_training))
        metrics_data[topic_id]["count_relevant_folders_total"] = len(folders_score_1)
        
        metrics_data[topic_id]["count_highly_relevant_folders_training"] = len(folders_score_3.intersection(folders_training))
        metrics_data[topic_id]["count_highly_relevant_folders_total"] = len(folders_score_3)
    
        run_results = folder_run_data.get(topic_id, {})

        run_results = sorted(run_results.items(), key=lambda x: x[1], reverse=True)

        top_5_folders_list = [item[0] for item in run_results[:5]]
        top_5_folders_set = set(top_5_folders_list)

        count_rel_in_top5 = len(folders_score_1.intersection(top_5_folders_set))
        count_high_in_top5 = len(folders_score_3.intersection(top_5_folders_set))

        # Adiciona ao JSON de métricas
        metrics_data[topic_id]["count_relevant_in_top5_model"] = count_rel_in_top5
        metrics_data[topic_id]["count_highly_relevant_in_top5_model"] = count_high_in_top5

    return metrics_data

def evaluateSearchResults(runFileName, folderQrelsFileName, boxQrelsFileName, outputFile, ecf, experiment_name, emb=False):
    measures = {'ndcg_cut', 'map', 'recip_rank', 'success'} # Generic measures for configuring a pytrec_eval evaluator
    measureNames = {'ndcg_cut_5': 'NDCG@5', 'map': '   MAP', 'recip_rank': '   MRR', 'success_1': '   S@1'} # Spedific measures for printing in pytrec_eval results

    with open(runFileName) as runFile, open(folderQrelsFileName) as folderQrelsFile, open(boxQrelsFileName) as boxQrelsFile:
        folderRun = {}

        if not emb:
            for line in runFile:
                topicId, folderId, rank, score, runName = line.split('\t')
                if topicId not in folderRun:
                    folderRun[topicId] = {}
                folderRun[topicId][folderId] = float(score)
        else:
            for line in runFile:
                topicId, folderId = line.strip().split('\t')
                if topicId not in folderRun:
                    folderRun[topicId] = {}
                folderRun[topicId][folderId] = 0

        # create json dump file from folderRun
        #with open(f'./results/topics/{experiment_name}_TopicsFolderRun.json', 'w') as f:
        #    json.dump(folderRun, f, indent=4)

        boxRun = makeBoxRun(folderRun)
        #with open(f'./results/topics/{NAME_EXPERIMENT}_TopicsBoxRun.json', 'w') as f:
        #    json.dump(boxRun, f, indent=4)

        folderQrels = {}
        for line in folderQrelsFile:
            topicId, unused, folderId, relevanceLevel = line.split('\t')
            if topicId not in folderQrels:
                folderQrels[topicId] = {}
            folderQrels[topicId][folderId] = int(relevanceLevel.strip())  # this deletes the \n at end of line
        folderEvaluator = pytrec_eval.RelevanceEvaluator(folderQrels, measures)
        folderTopicResults = folderEvaluator.evaluate(folderRun)  # replace run with folderQrels to see perfect evaluation measures

        if not emb:
            process_topics_enrichment_strict(folderTopicResults, ecf, folderQrelsFileName, folderRun)

        file_path = f'./all_runs/{experiment_name}_TopicsFolderMetrics.json'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(folderTopicResults, f, indent=4)

        boxQrels = {}
        for line in boxQrelsFile:
            topicId, unused, folderId, relevanceLevel = line.split('\t')
            if topicId not in boxQrels:
                boxQrels[topicId] = {}
            if folderId in boxQrels[topicId]:
                boxQrels[topicId][folderId] = max(boxQrels[topicId][folderId],int(relevanceLevel.strip()))  # strip() deletes the \n at end of line
            else:
                boxQrels[topicId][folderId] = int(relevanceLevel.strip())
        boxEvaluator = pytrec_eval.RelevanceEvaluator(boxQrels, measures)
        boxTopicResults = boxEvaluator.evaluate(boxRun) # replace run with qrels to see perfect evaluation measures
        
        #with open(f'./results/topics/{NAME_EXPERIMENT}_TopicsBoxMetrics.json', 'w') as f:
        #    json.dump(boxTopicResults, f, indent=4)

        pm='\u00B1'
        lines = []
        lines.append(f'          Folder          Box')
        for measure in measureNames.keys():
            folderMean, folderConf = stats(folderTopicResults, measure)
            boxMean, boxConf = stats(boxTopicResults, measure)
            lines.append(f'{measureNames[measure]}: {folderMean:.3f}{pm}{folderConf:.2f}    {boxMean:.3f}{pm}{boxConf:.2f}')
        # Save to file
        #with open(outputFile, "w") as out_f:
        #    for line in lines:
        #        out_f.write(line + "\n")
        # Also print to console if you want
        for line in lines:
            print(line)

## Notes: all random searches were removed
if __name__ == '__main__':
    # Set JAVA_HOME so that Terrier will work correctly
    # os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-22/"

    # Set global variables
    prefix = os.path.join(PROJECT_ROOT, 'data', 'raw')
    seq=0 # Controls index segments
    unix = True # Set to false for Windows, true for Unix.  This adapts the code to the locations where Terrier writes its index.
    model = 'terrier'

    #print("\n== Starting the SNC Translation...")
    #sncTranslation = load_snc_expansions('../src/SncTranslationV1.3.xlsx')
    #print("== Finished the SNC Translation\n")
    
    print("== Loading the Folder and Items Metadata...")
    folder_metadata_path = os.path.join(PROJECT_ROOT, 'data', 'folders_metadata', 'FoldersV1.2.json')
    items_metadata_path = os.path.join(PROJECT_ROOT, 'data', 'items_metadata', 'itemsV1.2.json')
    
    folderMetadata, items = load_folder_items_metadata(folder_metadata_path, items_metadata_path)
    # Print the Count of each SNC element in the FoldersMetadata:
    #print(count_snc(folderMetadata))
    print("== Loaded the Folder and Items Metadata\n")

    sys.exit()

    # Check Cossine Similarity between elements
    #checksim(items, 'A0001', 'A0002')

    # Plot Folders Dates Density
    #plotFolderDates(folderMetadata)

    # Test the official ECF and small versions of the official qrels files
    testing = True
    if (testing):
        searchFields = ['title']
        expansion = True
        print("== Loading Experiment Control File...")
        ecf = readExperimentControlFile('../ecf/formal_run/Ntcir18SushiOfficialExperimentControlFileV1.1.json')
        print(f"{len(ecf['ExperimentSets'])} Experiment Sets")
        print("== Loaded the Folder and Items Metadata\n")

        print(f"== Generating results for the searching fields: {''.join(searchFields)}...")
        results = generateSearchResults(ecf, searchFields, items, folderMetadata, "TD", expansion)
        print(f"== Generated results for all Experiment Sets.")
        
        print(f"== Writing Search Results...")
        writeSearchResults('./results/Ntcir18SushiOfficialResultsV1.1.tsv', results, '4-topic-test')
        print(f"== Search Results written.")

        print("== Evaluating Search Results...")
        qrels_folder = '../qrels/formal-run-qrels/formal-folder-qrel.txt'
        qrels_box = '../qrels/formal-run-qrels/formal-box-qrel.txt'
        evaluateSearchResults('./results/Ntcir18SushiOfficialResultsV1.1.tsv', qrels_folder, qrels_box, './results/metrics/OfficialResultsTopicsResultsEvaluation.txt', ecf, "FormalRun")
        print("== Search results evaluated.")