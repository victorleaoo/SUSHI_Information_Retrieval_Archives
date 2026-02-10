import os
import statistics
import csv

# --- Existing Collection Functions ---

def sortLongest(my_dict):
    dict_lengths = {key: len(value) for key, value in my_dict.items()}
    sorted_keys = sorted(dict_lengths, key=lambda k: dict_lengths[k], reverse=True)
    sorted_dict = {key: my_dict[key] for key in sorted_keys}
    return sorted_dict

def getSushiFiles(dir):
    fullCollection = {}
    print("Starting walk of", dir)
    try:
        for box in os.listdir(dir):
            box_path = os.path.join(dir, box)
            if not os.path.isdir(box_path): continue
            
            fullCollection[box] = {}
            for folder in os.listdir(box_path):
                folder_path = os.path.join(box_path, folder)
                if not os.path.isdir(folder_path): continue
                
                fullCollection[box][folder] = []
                for file in os.listdir(folder_path):
                    fullCollection[box][folder].append(file)
                    
        for box in fullCollection:
            fullCollection[box] = sortLongest(fullCollection[box])
            
        print("Finished getting documents")
        return fullCollection
    except FileNotFoundError:
        print(f"Error: Directory '{dir}' not found.")
        return {}

def print_detailed_stats(label, data_tuples):
    """
    Calculates stats (Min, Max, Mean, Median, StdDev) and prints specific examples.
    data_tuples: List of (Name, Count) -> [('Box1', 10), ('Box2', 5), ...]
    """
    if not data_tuples:
        print(f"--- {label} ---\nNo data available.\n")
        return

    # Extract just the counts for math
    counts = [item[1] for item in data_tuples]
    
    min_val = min(counts)
    max_val = max(counts)
    mean_val = statistics.mean(counts)
    median_val = statistics.median(counts)
    stdev_val = statistics.stdev(counts) if len(counts) > 1 else 0.0
    
    # Identify the names associated with Min and Max
    min_examples = [item[0] for item in data_tuples if item[1] == min_val]
    max_examples = [item[0] for item in data_tuples if item[1] == max_val]

    print(f"--- {label} ---")
    print(f"  Mean:   {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  StdDev: {stdev_val:.2f}")
    
    # Print Min with examples
    min_str = ", ".join(min_examples[:3]) 
    if len(min_examples) > 3: 
        min_str += f" ... ({len(min_examples)-3} others)"
    print(f"  Min:    {min_val}  [Found in: {min_str}]")

    # Print Max with examples
    max_str = ", ".join(max_examples[:3]) 
    if len(max_examples) > 3: 
        max_str += f" ... ({len(max_examples)-3} others)"
    print(f"  Max:    {max_val}  [Found in: {max_str}]")
    print("")

def analyze_collection_detailed(fullCollection):
    if not fullCollection:
        print("Collection is empty.")
        return

    # Data structures to hold (Name, Count) tuples
    folders_per_box_data = []
    docs_per_box_data = []
    docs_per_folder_data = []

    total_folders = 0
    total_docs = 0

    for box, folders in fullCollection.items():
        # 1. Folders per Box
        folder_count = len(folders)
        folders_per_box_data.append((box, folder_count))
        total_folders += folder_count

        box_doc_total = 0
        for folder, docs in folders.items():
            # 2. Documents per Folder
            doc_count = len(docs)
            # Use "Box/Folder" as name to be unique
            docs_per_folder_data.append((f"{box}/{folder}", doc_count))
            box_doc_total += doc_count
            total_docs += doc_count
        
        # 3. Documents per Box
        docs_per_box_data.append((box, box_doc_total))

    # Print Totals
    print("=== Collection Totals ===")
    print(f"Total Boxes:     {len(fullCollection)}")
    print(f"Total Folders:   {total_folders}")
    print(f"Total Documents: {total_docs}")
    print("=========================\n")

    # Print Detailed Statistics
    print_detailed_stats("Folders per Box", folders_per_box_data)
    print_detailed_stats("Documents per Box", docs_per_box_data)
    print_detailed_stats("Documents per Folder", docs_per_folder_data)

def analyze_qrels(qrel_paths):
    """
    Analyzes Box, Folder, and Document QREL files.
    qrel_paths: dict {'Box': 'path', 'Folder': 'path', 'Document': 'path'}
    """
    for q_type, path in qrel_paths.items():
        print(f"\n##########################################")
        print(f"### QREL Analysis: {q_type}")
        print(f"##########################################\n")
        
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        # Storage for counts per topic
        # Structure: {'TopicID': count}
        total_relevant = {} # 1 or 3
        relevant_1 = {}     # 1 only
        relevant_3 = {}     # 3 only
        
        # Set of all topics seen (to handle 0 counts correctly if needed, 
        # though standard practice usually analyzes distribution of positive hits)
        topics_seen = set()

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4: continue
                
                topic = parts[0]
                try:
                    label = int(parts[3])
                except ValueError:
                    continue # Skip header or malformed lines

                topics_seen.add(topic)
                
                # Filter: Only consider 1 and 3
                if label in [1, 3]:
                    total_relevant[topic] = total_relevant.get(topic, 0) + 1
                    
                    if label == 1:
                        relevant_1[topic] = relevant_1.get(topic, 0) + 1
                    elif label == 3:
                        relevant_3[topic] = relevant_3.get(topic, 0) + 1

        # Convert to list of tuples for the stats function: [(Topic, Count), ...]
        # We perform a check: if a topic was seen in the file but has no relevant items,
        # it won't be in the dictionaries. We explicitly set those to 0 to represent "0 relevant items found".
        
        def prepare_data(count_dict, all_topics):
            data = []
            for t in all_topics:
                data.append((t, count_dict.get(t, 0)))
            return data

        data_total = prepare_data(total_relevant, topics_seen)
        data_1 = prepare_data(relevant_1, topics_seen)
        data_3 = prepare_data(relevant_3, topics_seen)

        # 1. General View (1 + 3)
        print(">>> GENERAL VIEW (Labels 1 & 3)")
        print_detailed_stats(f"Relevant {q_type}s per Topic", data_total)
        
        # 2. Specific View (1 vs 3)
        print(">>> SPECIFIC VIEW")
        print_detailed_stats(f"Label 1 (Relevant) {q_type}s per Topic", data_1)
        print_detailed_stats(f"Label 3 (Highly Relevant) {q_type}s per Topic", data_3)


# --- Execution Example ---

# 1. Run Collection Analysis
# fullCollection = getSushiFiles('./raw/')
# analyze_collection_detailed(fullCollection)

# 2. Run QREL Analysis
qrel_files = {
    'Box': '../qrels/formal-box-qrel.txt',
    'Folder': '../qrels/formal-folder-qrel.txt',
    'Document': '../qrels/formal-document-qrel.txt'
}
analyze_qrels(qrel_files)