import os
import math
import pytrec_eval
import json

# Setting base path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..'))

def writeSearchResults(fileName, results, runName):
    with open(fileName, 'w') as f:
        for topic in results:
            for i in range(len(topic['RankedList'])):
                print(f'{topic["Id"]}\t{topic["RankedList"][i]}\t{i+1}\t{1/(i+1):.4f}\t{runName}', file=f)
    f.close()

def createFolderToBoxMap():
    boxMap = {}
    with open(os.path.join(base_dir, 'data', 'folders_metadata', 'FoldersV1.2.json')) as foldersFile:
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

def evaluateSearchResults(runFileName, folderQrelsFileName, boxQrelsFileName, outputFile):
    #print(pytrec_eval.supported_measures)
    measures = {'ndcg_cut', 'map', 'recip_rank', 'success'} # Generic measures for configuring a pytrec_eval evaluator
    measureNames = {'ndcg_cut_5': 'NDCG@5', 'map': '   MAP', 'recip_rank': '   MRR', 'success_1': '   S@1'} # Spedific measures for printing in pytrec_eval results

    with open(runFileName) as runFile, open(folderQrelsFileName) as folderQrelsFile, open(boxQrelsFileName) as boxQrelsFile:
        folderRun = {}

        for line in runFile:
            topicId, folderId, rank, score, runName = line.split('\t')
            if topicId not in folderRun:
                folderRun[topicId] = {}
            folderRun[topicId][folderId] = float(score)

        # create json dump file from folderRun
        with open('results/topics/TopicsFolderRun.json', 'w') as f:
            json.dump(folderRun, f, indent=4)

        boxRun = makeBoxRun(folderRun)
        with open('results/topics/TopicsBoxRun.json', 'w') as f:
            json.dump(boxRun, f, indent=4)

        folderQrels = {}
        for line in folderQrelsFile:
            topicId, unused, folderId, relevanceLevel = line.split('\t')
            if topicId not in folderQrels:
                folderQrels[topicId] = {}
            folderQrels[topicId][folderId] = int(relevanceLevel.strip())  # this deletes the \n at end of line
        folderEvaluator = pytrec_eval.RelevanceEvaluator(folderQrels, measures)
        folderTopicResults = folderEvaluator.evaluate(folderRun)  # replace run with folderQrels to see perfect evaluation measures

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

        pm='\u00B1'
        lines = []
        lines.append(f'          Folder          Box')
        for measure in measureNames.keys():
            folderMean, folderConf = stats(folderTopicResults, measure)
            boxMean, boxConf = stats(boxTopicResults, measure)
            lines.append(f'{measureNames[measure]}: {folderMean:.3f}{pm}{folderConf:.2f}    {boxMean:.3f}{pm}{boxConf:.2f}')
        # Save to file
        with open(outputFile, "w") as out_f:
            for line in lines:
                out_f.write(line + "\n")
        # Also print to console if you want
        for line in lines:
            print(line)