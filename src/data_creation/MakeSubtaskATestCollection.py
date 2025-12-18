import os
import json
import pandas as pd
import random

RANDOM_SEED = 150

def sortLongest(my_dict):
    dict_lengths = {key: len(value) for key, value in my_dict.items()}
    sorted_keys = sorted(dict_lengths, key=lambda k: dict_lengths[k], reverse=True)
    sorted_dict = {key: my_dict[key] for key in sorted_keys}
    return sorted_dict

def getSushiFiles(dir):
    fullCollection = {}
    print("Starting walk of", dir)
    for box in os.listdir(dir):
        #print(f'Reading SUSHI collection box {box}')
        fullCollection[box] = {}
        for folder in os.listdir(os.path.join(dir,box)):
            fullCollection[box][folder] = []
#            print(f'Read box {box}, folder {folder}')
            for file in os.listdir(os.path.join(dir,box,folder)):
#                print(f'Read file {os.path.join(dir,box,folder,file)}')
                fullCollection[box][folder].append(file)
    for box in fullCollection:
        fullCollection[box] = sortLongest(fullCollection[box])
    print("Finished getting documents")
    return fullCollection

def full_trainingset_topicset(queries, fullCollection):
    def selectAllTraining(fullCollection):
        trainingSet = []

        for box in fullCollection:
            for folder in fullCollection[box]:
                for file in fullCollection[box][folder]:
                    trainingSet.append(f'{box}/{folder}/{file}')

        trainingSet.sort() 
        return trainingSet
    
    trainingSets = []
    topicSets = []

    if queries:
        print(f'Total of {len(queries)} unique queries are available.\n')
        topicSets.append(queries) 

        trainingSets.append(selectAllTraining(fullCollection))
        
        print(f'Topic Set 0 (All Topics) with length {len(topicSets[0])}')
        print(f'Training Set 0 (All Data) with length {len(trainingSets[0])}')
        
        print(f"\nLen of trainingSets: {len(trainingSets)} - Len of topicSets: {len(topicSets)}")
    else:
        print('No topics list given')
        exit(-1)
        
    return topicSets, trainingSets


def selectUniformTraining(fullCollection, docsPerBox, random_seed):
    trainingSet = []
    trainingFiles = []
    max = 300

    count_f = 0

    for box in fullCollection:
        folderDocs = [0]*max
        total = 0
        for j in range(docsPerBox): # 5 documents per box
            # if the box has less than 5 folders, some folders will have more documents selected
            for i in range(min(len(fullCollection[box]),docsPerBox,max)): # select the min between the len of folders in the box, 5 and 300
                if total<docsPerBox:
                    folderDocs[i] += 1
                    total += 1
        #print(folderDocs)

        i = 0

        random.seed(random_seed)

        # RANDOM FOLDERS, BEFORE IT WAS LINEAR
        folders_list = list(fullCollection[box].keys())
        random.shuffle(folders_list)

        for folder in folders_list:
            valid_candidates = [doc for doc in fullCollection[box][folder] if doc not in trainingFiles]
            num_to_pick = min(folderDocs[i], len(valid_candidates))
            if num_to_pick > 0:
                # random selection of valid documents candidates (not yet choosen)
                selected = random.sample(valid_candidates, num_to_pick)
                
                for candidate in selected:
                    trainingFiles.append(candidate)
                    trainingSet.append(box+'/'+folder+'/'+candidate)
            i+=1
            count_f += 1
        trainingSet.sort()

    return trainingSet

def setupEcf(queries, fullCollection, random_seed):
    trainingSets = []

    if queries:
        print(f'Total of {len(queries)} unique queries are available.\n')
        random.seed(random_seed)
        random.shuffle(queries)

        topicSets = [queries[0:15], queries[15:30], queries[30:]]

        #print(topicSets)

        for i in range(len(topicSets)):
            trainingSets.append(selectUniformTraining(fullCollection, 5, random_seed))
            print(f'Topic Set {i} with length {len(topicSets[i])}')
            print(f'Training Set {i} with length {len(trainingSets[i])}')
        print(f"\nLen of trainingSets: {len(trainingSets)} - Len of topicSets: {len(topicSets)}")
    else:
        print('No topics list given')
        exit(-1)
    return topicSets, trainingSets

def writeJson(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def writeEcf(fileName, experimentName, trainingSets, topicSets, topicPrefix, firstTopicNumber):
    ecf = {}
    ecf['ExperimentName'] = experimentName
    ecf['ExperimentSets'] = []
    if len(trainingSets) != len(topicSets):
        print(f'Mismatch between {trainingSets} Training Sets and {topicSets} Topic Sets; Aborted')
        exit(-1)
    for set in range(len(trainingSets)):
        ecf['ExperimentSets'].append({})
        ecf['ExperimentSets'][set]['TrainingDocuments'] = trainingSets[set]
        ecf['ExperimentSets'][set]['Topics'] = {}
        for topic in range(len(topicSets[set])):
            topicId = topicSets[set][topic]['ID']
            ecf['ExperimentSets'][set]['Topics'][topicId] = topicSets[set][topic]
            topicSets[set][topic]['ID'] = topicId
        firstTopicNumber += len(topicSets[set])
    writeJson(ecf, fileName)
    return topicSets

if __name__ == '__main__':
    fullCreation = False
    fullCollection = getSushiFiles('/Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/data/raw/')
    
    with open("./topics_output.txt", 'r', encoding='utf-8') as file:
        queries = list(json.load(file).values())

    if fullCreation:
        topicSets, trainingSets = full_trainingset_topicset(queries, fullCollection)
        ecf_path = f'/Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/ecf/random_generated/ECF_ALL_TRAINING_SET.json'
        topicSets = writeEcf(ecf_path, 'All Docs in TrainingSet', trainingSets, topicSets, 'FULL_TOPICS', 1)
    else:
        topicSets, trainingSets = setupEcf(queries, fullCollection) #seting dryrun to true uses manually selected queries
        ecf_path = f'/Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/ecf/random_generated/ECF_RANDOM_{RANDOM_SEED}.json'
        topicSets = writeEcf(ecf_path, f'ECF w/ Random Seed {RANDOM_SEED}', trainingSets, topicSets, 'TEST', 1)