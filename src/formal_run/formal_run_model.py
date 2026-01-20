import os
import json
import shutil
import pyterrier as pt # For this to work you should pip install python-terrier
import pandas as pd
import sys
import re

# Setting base path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SNCLabelTranslate import translate_snc_label
from SUSHI_Information_Retrieval_Archives.src.lastest_runs.SubtaskAEvaluation import writeSearchResults, evaluateSearchResults

CONTROL_FILE_PREFIX = os.path.join(base_dir, 'ecf', 'formal_run', 'Ntcir18SushiOfficialExperimentControlFileV1.1.json')

def readExperimentControlFile(fileName):
    with open(fileName) as ecfFile:
        ecf = json.load(ecfFile)
    return ecf

def trainTerrierModel(trainingDocs, searchFields):
    global seq # Used to control creation of a separate index for each training set
    global unix
    global prefix

    trainingSet=[]

    #### open items json file
    with open(os.path.join(base_dir, 'data', 'items_metadata', 'itemsV1.2.json')) as itemsFile:
        items = json.load(itemsFile)

    # Build the data structure that Terrier will index (list of dicts, one per indexed item)
    for trainingDoc in trainingDocs:
        # Read the box/folder/file directory structure
        sushiFile = trainingDoc[-10:-4] # This extracts the file name and ignores the box and folder labels which we will get from the metadata
        file = sushiFile

        folder = items[file]["Sushi Folder"]
        box = items[file]["Sushi Box"]

        # SNC Label Translation
        xls = pd.ExcelFile('../SncTranslationV1.3.xlsx')
        sncExpansion = xls.parse(xls.sheet_names[0])

        brownLabel = items[file]["Brown Folder Name"] # Not used in indexing
        naraLabel = items[file]["NARA Folder Name"]   # Not used in indexing

        label = translate_snc_label(naraLabel, brownLabel, sushiFile, folder, sncExpansion)

        brownTitle = items[file]["Brown Title"]
        naraTitle = items[file]["NARA Title"]

        if not isinstance(brownTitle, float): # If there's a Brown title, use its
            title = brownTitle
        else:
            start = naraTitle.find('Concerning')
            if start != -1:
                naraTitle = naraTitle[start+11:]
            
            end1 = naraTitle.rfind(':')
            end2 = naraTitle.rfind('(')
            end = min(end1,end2)
            if end != -1:
                naraTitle = naraTitle[:end]

            title = naraTitle
        
        ocr = items[file]["ocr"][0]

        summary = items[file]["summary"]

        trainingSet.append({'docno': file, 'folder': folder, 'box': box, 'title': title, 'ocr': ocr, 'folderlabel': label, 'summary': summary})

    # Create the Terrier index for this training set and then return a Terrier retriever for that index
    seq += 1 # We create one Terrier index per training set
    indexDir = prefix + 'terrierindex/'+str(seq) # Be careful here -- this directory and all its contents will be deleted!
    if 'index' in indexDir and os.path.isdir(indexDir):
        print(f'Deleting prior index {indexDir}')
        shutil.rmtree(indexDir) # This is required because Terrier fails to close its index on completion

    if not pt.java.started(): 
        pt.java.init()

    indexer = pt.IterDictIndexer(indexDir, 
                                meta={'docno': 20, 'folder':20, 'box': 20, 'title':16384, 'ocr':16384, 'folderlabel': 1024, 'summary': 16384}, 
                                meta_reverse=['docno', 'folder', 'box'],
                                overwrite=True,
                                text_attrs=searchFields
                                )
    indexref = indexer.index(trainingSet)
    index = pt.IndexFactory.of(indexref)

    BM25 = pt.terrier.Retriever(index, 
                                wmodel="BM25",
                                metadata=['docno', 'folder', 'box', 'title', 'ocr', 'folderlabel', 'summary'],
                                num_results=1000
                               )
    return BM25

def trainModel(trainingDocuments, searchFields):
    global seq
    print(f'Training Called, preparing index for experiment set {seq+1}')
    return trainTerrierModel(trainingDocuments, searchFields)

# 1. Recebe o dicionário 'output_data' como argumento
def saveSearchText(query, result, topicId, output_data):
    # Removemos o 'global', agora usamos o argumento passado
    
    # Concatena os campos de texto
    result['search_text'] = (
        result['title'].fillna('') + " " + 
        result['ocr'].fillna('') + " " + 
        result['folderlabel'].fillna('') + " " + 
        result['summary'].fillna('')
    ).str.strip()

    ranked_df = result[['folder', 'score', 'search_text']].copy()
    # Mantém o documento com maior score da pasta para representar o texto
    ranked_df.drop_duplicates(subset='folder', keep='first', inplace=True)
    
    # Pega top 10 e converte para dicionário
    top_10 = ranked_df.head(10).to_dict(orient='records')

    # Salva no dicionário que veio lá da função principal
    output_data[topicId] = {
        "query": query,
        "top_10_results": top_10
    }

# 2. Recebe 'set_output_data' e repassa para saveSearchText
def terrierSearch(query, engine, topicId, set_output_data):
    if not pt.java.started(): 
        pt.java.init()

    query = re.sub(r'[^a-zA-Z0-9\s]', '', query) 
    result = engine.search(query)

    # Passamos o dicionário adiante para ser preenchido
    saveSearchText(query, result, topicId, set_output_data)

    rankedList = result['folder']
    rankedList.drop_duplicates(inplace=True)
    
    return rankedList.tolist()

def generateSearchResults(ecf, searchFields):
    results = []
    i = 0
    
    # Garante que o diretório de saída existe
    if not os.path.exists("./results/exir/"):
        os.makedirs("./results/exir/")

    # Usamos enumerate para saber qual é o número do Experiment Set atual (0, 1, 2...)
    for set_idx, experimentSet in enumerate(ecf['ExperimentSets']):
        
        # Cria o dicionário novo para ESTE set específico
        set_output_data = {}
        
        index = trainModel(experimentSet['TrainingDocuments'], searchFields)
        topics = list(experimentSet['Topics'].keys())
        
        for j in range(len(topics)):
            results.append({})
            results[i]['Id'] = topics[j]
            
            query = experimentSet['Topics'][topics[j]]['TITLE']
            
            # Passamos set_output_data como argumento aqui
            rankedFolderList = terrierSearch(query, index, topics[j], set_output_data)
            
            results[i]['RankedList'] = rankedFolderList
            i += 1
        
        # Salvamos usando set_idx + 1 para ficar "set_1", "set_2" corretamente
        filepath = f"./results/exir/experiment_set_{set_idx + 1}_explainable.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(set_output_data, f, indent=4, ensure_ascii=False)

    return results

def writeSearchResults(fileName, results, runName):
    with open(fileName, 'w') as f:
        for topic in results:
            for i in range(len(topic['RankedList'])):
                print(f'{topic["Id"]}\t{topic["RankedList"][i]}\t{i+1}\t{1/(i+1):.4f}\t{runName}', file=f)
    f.close()

if __name__ == '__main__':
    # Params
    unix = True
    seq = 0
    prefix = '/Users/victorleao/mestrado/sushi_docs/SUSHI_Information_Retrieval_Archives/src/formal_run/'

    # Run experiment
    searchFields = ['title', 'ocr', 'folderlabel', 'summary']
    ecf = readExperimentControlFile(CONTROL_FILE_PREFIX)
    results = generateSearchResults(ecf, 
                                    searchFields)
    
    writeSearchResults('./results/topics/FormalRunTopicsResults.tsv', 
                       results, 
                       'Baseline-TOFS')
    
    evaluateSearchResults('./results/topics/FormalRunTopicsResults.tsv', 
                          os.path.join(base_dir, 
                                       'qrels', 'formal-run-qrels', 'formal-folder-qrel.txt'), # folder qrels
                          os.path.join(base_dir, 
                                       'qrels', 'formal-run-qrels', 'formal-box-qrel.txt'), # box qrels
                          './results/metrics/FormalRunTopicsResultsEvaluation.txt'
                         )