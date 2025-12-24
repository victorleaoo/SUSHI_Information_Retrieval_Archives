import json
import shutil
import os

from metric_generator import metric_generator

from lastest_runs.run_refactor import load_snc_expansions, load_folder_items_metadata, readExperimentControlFile, generateSearchResults, writeSearchResults, evaluateSearchResults

from data_creation.MakeSubtaskATestCollection import getSushiFiles, setupEcf, writeEcf

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def saving_folder_name(queryFields, searchFields, expansion):
    field_map = {
        'folderlabel': 'F', 
        'ocr': 'O', 
        'summary': 'S', 
        'title': 'T'
    }

    search_field_name = ""

    for field in searchFields:
        search_field_name += field_map[field]

    expansion_code = "EX" if expansion else "NEX"

    query_fields_code = queryFields

    return f"{search_field_name}_{expansion_code}_{query_fields_code}"

if __name__ == '__main__':
    print("== Loading the Folder and Items Metadata...")
    folder_metadata_path = os.path.join(PROJECT_ROOT, 'data', 'folders_metadata', 'FoldersV1.2.json')
    items_metadata_path = os.path.join(PROJECT_ROOT, 'data', 'items_metadata', 'itemsV1.2.json')
    folderMetadata, items = load_folder_items_metadata(folder_metadata_path, items_metadata_path)
    print("== Loaded the Folder and Items Metadata\n")

    queryFields = ["T"] # ["T", "TD", "TDN"]
    searchFields = [['title'], ['ocr']] # [['title'], ['ocr'], ['folderlabel'], ['summary'], ['title', 'ocr', 'folderlabel', 'summary']]
    expansion = True
    random = False

    allDocs = True

    for searchField in searchFields:
        for queryField in queryFields:
            if random:
                random_seed_list = [1, 42, 100, 300, 333, 777, 999, 2025, 6159, 12345,
                                    19865, 53819, 56782, 62537, 72738, 75259, 81236, 91823, 98665, 98765, 99009,
                                    120302, 123865, 170302, 180803, 5122025, 12052024, 12052025, 99000011, 999777333] # 30 random generations
                
                for random_seed in random_seed_list:
                    print(f"\n========= ITERATION FOR SEED {random_seed} =========\n")

                    print("== Loading Experiment Control File...")

                    fullCollection = getSushiFiles(os.path.join(PROJECT_ROOT, 'data', 'raw'))
                    with open(os.path.join(PROJECT_ROOT, "src", "data_creation", "topics_output.txt"), 'r', encoding='utf-8') as file:
                        queries = list(json.load(file).values())

                    topicSets, trainingSets = setupEcf(queries, fullCollection, random_seed)
                    ecf_path = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated', f'ECF_RANDOM_{random_seed}.json')
                    topicSets = writeEcf(ecf_path, f'ECF w/ Random Seed {random_seed}', trainingSets, topicSets, 'TEST', 1)

                    ecf = readExperimentControlFile(ecf_path)
                    print(f"{len(ecf['ExperimentSets'])} Experiment Sets")
                    print("== Loaded the Folder and Items Metadata\n")

                    print(f"== Generating results for the searching fields: {''.join(searchField)}...")
                    results = generateSearchResults(ecf, searchField, items, folderMetadata, queryFields=queryField, expansion=expansion)
                    print(f"== Generated results for all Experiment Sets.")
                    
                    print(f"== Writing Search Results...")
                    writeSearchResults(os.path.join(PROJECT_ROOT, 'results', 'Ntcir18SushiOfficialResultsV1.1.tsv'), results, '4-topic-test')
                    print(f"== Search Results written.")

                    print("== Evaluating Search Results...")
                    qrels_folder = os.path.join(PROJECT_ROOT, 'qrels', 'formal-run-qrels', 'formal-folder-qrel.txt')
                    qrels_box = os.path.join(PROJECT_ROOT, 'qrels', 'formal-run-qrels', 'formal-box-qrel.txt')
                    evaluateSearchResults(
                        os.path.join(PROJECT_ROOT, 'results', 'Ntcir18SushiOfficialResultsV1.1.tsv'),
                        qrels_folder,
                        qrels_box,
                        os.path.join(PROJECT_ROOT, 'results', 'metrics', 'OfficialResultsTopicsResultsEvaluation.txt'),
                        ecf,
                        f"{saving_folder_name(queryField, searchField, expansion)}/Random{random_seed}"
                    )
                    print(f'======= Saved Random{random_seed} -> {saving_folder_name(queryField, searchField, expansion)} =======')
                    print("== Search results evaluated.")
            else:
                if allDocs:
                    name_run = f"AllTrainingDocuments{queryField}"
                    ecf = readExperimentControlFile(os.path.join(PROJECT_ROOT, 'ecf', 'random_generated', 'ECF_ALL_TRAINING_SET.json'))
                    print(f"{len(ecf['ExperimentSets'])} Experiment Sets")
                    print("== Loaded the Folder and Items Metadata\n")
                else:
                    name_run = f"FormalRun{queryField}"
                    ecf = readExperimentControlFile(os.path.join(PROJECT_ROOT, 'ecf', 'formal_run', 'Ntcir18SushiOfficialExperimentControlFileV1.1.json'))
                    print(f"{len(ecf['ExperimentSets'])} Experiment Sets")
                    print("== Loaded the Folder and Items Metadata\n")
                    
                print(f"== Generating results for the searching fields: {''.join(searchField)}...")
                results = generateSearchResults(ecf, searchField, items, folderMetadata, queryFields=queryField, expansion=expansion)
                print(f"== Generated results for all Experiment Sets.")
                
                print(f"== Writing Search Results...")
                writeSearchResults('./results/Ntcir18SushiOfficialResultsV1.1.tsv', results, '4-topic-test')
                print(f"== Search Results written.")

                print("== Evaluating Search Results...")
                qrels_folder = os.path.join(PROJECT_ROOT, 'qrels', 'formal-run-qrels', 'formal-folder-qrel.txt')
                qrels_box = os.path.join(PROJECT_ROOT, 'qrels', 'formal-run-qrels', 'formal-box-qrel.txt')
                evaluateSearchResults(
                    './results/Ntcir18SushiOfficialResultsV1.1.tsv',
                    qrels_folder,
                    qrels_box,
                    './results/metrics/OfficialResultsTopicsResultsEvaluation.txt',
                    ecf,
                    f"{saving_folder_name(queryField, searchField, expansion)}/AllTrainingDocuments"
                )
                print(f"======== {saving_folder_name(queryField, searchField, expansion)}/AllTrainingDocuments ========")
                print("== Search results evaluated.")

            metric_generator(random)