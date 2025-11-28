import os

def loadValidFolders():
    dir='d:/sushi/sushi-files/'
    folders = {}
    for box in os.listdir(dir):
        for folder in os.listdir(os.path.join(dir, box)):
            folders[folder] = []
            for file in os.listdir(os.path.join(dir, box, folder)):
                folders[folder].append(file)
    return(folders)

def validateRunFile(runFileName, validFolders):
    with open(runFileName) as runFile:
        priorTopicId = ''
        firstRunName = ''
        items = []
        ranks = []
        scores = []
        for line in runFile:
            elements = line.split('\t')
            if len(elements) != 5:
                print(f'Invalid run file format: All lines must have 5 tab-separated values')
                return False
            topicId, folderId, rankString, scoreString, runName = elements
            if not((len(topicId)==9 and 'T18-' in topicId and str.isdigit(topicId[-5:])) or len(topicId)==15 and 'T18DryRun' in topicId and str.isdigit(topicId[-5:])):
                print(f'Topic ID {topicId} is not valid for a submitted run')
                return False
            if folderId not in validFolders:
                print(f'Folder Name {folderId} for Topic {topicId} is not valid for this test collection')
                return False
            if not str.isdigit(rankString):
                print('Rank {rankString} for Topic {topicId} Folder {folderId} can not be read as an integer')
                return False
            rank = int(rankString)
            if rank<1 or rank>1000:
                print('Rank {rank} not between 1 and 1000 for Topic {priorTopicId} Folder {folderId}')
            try:
                score = float(scoreString)
            except ValueError:
                print(f'Score {score} for Topic {topicId} Folder {folderId} can not be read as a floating point number')
                return folderId
            if topicId == priorTopicId:
                items.append(folderId)
                ranks.append(rank)
                scores.append(score)
            else:
                for i in range(len(items)):
                    if i<len(items) and items[i] in items[i+1:]:
                        print(f'Folder Name {folderId} appears more than once in the ranked list for Topic {priorTopicId}')
                        return False
                    if i<len(ranks)-1 and ranks[i+1]!=ranks[i]+1:
                        print(f'Ranks are not in ascending order for Topic {priorTopicId}')
                        return False
                    if i<len(ranks)-1 and scores[i]<scores[i+1]:
                        print(f'Scores {scores[i]} and {scores[i+1]} are not in nonincreasing order for Topic {priorTopicId}')
                        return False
                items = [folderId]
                ranks = [rank]
                scores= [score]
                priorTopicId = topicId
            if firstRunName=='':
                firstRunName = runName
                if len(runName)<2 or len(runName)>20:
                    print('Run name is not between 2 and 20 characters')
                    return False
            if runName != firstRunName:
                print(f'Run name changes from {firstRunName.strip()} to {runName.strip()} in Topic {topicId} Folder {folderId}')
                return False
        return True

if __name__ == '__main__':
    folders = loadValidFolders()
    runFileName = 'd:/sushi/Ntcir18SushiDryRunResultsV1.1.tsv'
    valid = validateRunFile(runFileName, folders)
    if valid:
        print()
        print(f'Run file {runFileName} passes all validation checks')
    else:
        exit(-1)
