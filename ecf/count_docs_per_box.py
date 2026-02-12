import json
from collections import Counter

# Load the JSON data
with open('random_generated/ECF_3perbox_Random_Seed_42.json', 'r') as f:
    data = json.load(f)

box_counts = Counter()

# Iterate through the ExperimentSets
for experiment_set in data.get("ExperimentSets", []):
    for doc_path in experiment_set.get("TrainingDocuments", []):
        parts = doc_path.split('/')
        
        if parts:
            box_id = parts[0]
            box_counts[box_id] += 1

# Print the results
print("File counts per box:")
sum = 0
for box, count in box_counts.items():
    print(f"{box}: {count}")
    sum += count
print(sum)