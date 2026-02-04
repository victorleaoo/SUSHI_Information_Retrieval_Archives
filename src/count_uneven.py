import json
from collections import Counter

def count_boxes_in_ecf(data):
    # 1. Access the list of document paths
    docs = data["ExperimentSets"][0]["TrainingDocuments"]
    
    # 2. Extract the Box ID from each path
    # Path format is "BoxID/FolderID/FileID.pdf"
    # We split by '/' and take the first element
    box_ids = [path.split('/')[0] for path in docs]
    
    # 3. Count frequencies
    counts = Counter(box_ids)
    
    # 4. Sort by count (Descending)
    sorted_counts = counts.most_common()
    
    print(f"{'Box ID':<10} | {'Count'}")
    print("-" * 20)
    for box, count in sorted_counts:
        print(f"{box:<10} | {count}")

# Load and run
with open("../ecf/random_generated/ECF_UNEVEN_Random_Seed_99000011.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

count_boxes_in_ecf(data)