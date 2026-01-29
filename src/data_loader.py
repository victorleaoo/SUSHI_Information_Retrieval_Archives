import os
import json
import random

class DataLoader:
    """
    Central data management utility for the experiment pipeline.
    
    Responsible for handling all file input/output operations, including loading metadata, mapping the physical directory structure of the dataset, and generating "Experimental Collection Formats" (ECF).

    Attributes:
        project_root (str): The absolute path to the project's root directory.
    """
    def __init__(self, 
                 project_root):
        self.project_root = project_root
        self.folder_metadata_path = os.path.join(project_root, 'data', 'folders_metadata', 'FoldersV1.3.json')
        self.items_metadata_path = os.path.join(project_root, 'data', 'items_metadata', 'itemsV1.2.json')
        self.sushi_files_path = os.path.join(project_root, 'data', 'raw')
        self.topics_path = os.path.join(project_root, "src", "data_creation", "topics_output.txt")
        self.all_docs_ecf_path = os.path.join(project_root, 'ecf', 'random_generated', 'ECF_ALL_TRAINING_SET.json')
        
        self.items = self._load_json(self.items_metadata_path)
        self.folder_metadata = self._load_json(self.folder_metadata_path)
        self.full_collection = self._build_full_collection()

    def _load_json(self, path):
        """
        Helper method to safely open and parse a JSON file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_full_collection(self):
        """
        Scans the raw data directory to build a complete hierarchy of the collection.

        Iterates through every box and folder in the `data/raw` path to collect filenames. It sorts folders within each box by the number of files (descending) to optimize processing order during sampling.

        Returns:
            dict: The collection structure in the format: {box_id: {folder_id: [file_names]}}.
        """
        collection = {}
        for box in os.listdir(self.sushi_files_path):
            box_path = os.path.join(self.sushi_files_path, box)
            if not os.path.isdir(box_path): continue
            
            collection[box] = {}
            for folder in os.listdir(box_path):
                folder_path = os.path.join(box_path, folder)
                if not os.path.isdir(folder_path): continue
                
                files = os.listdir(folder_path)
                collection[box][folder] = files
        
        # Sort folders by number of files (longest first)
        for box in collection:
            sorted_folders = sorted(collection[box], key=lambda k: len(collection[box][k]), reverse=True)
            collection[box] = {k: collection[box][k] for k in sorted_folders}
            
        return collection

    def get_topics(self):
        """
        Loads the standardized list of research topics (queries) for the experiments.

        Returns:
            list[dict]: A list where each dictionary represents a topic (containing keys like ID, Title, Description, Narrative).
        """
        with open(self.topics_path, 'r', encoding='utf-8') as f:
            return list(json.load(f).values())

    def create_random_ecf(self, seed, sampling='uniform', max_docs=300, docs_per_box=5):
        """
        Generates a randomized "Experimental Collection Format" (ECF) object.

        Args:
            seed (int): The random seed to ensure the selection is reproducible.
            sampling (str, optional): The strategy for selecting documents. Defaults to 'uniform'.
            max_docs (int, optional): The hard limit on documents to sample per box. Defaults to 300.
            docs_per_box (int, optional): The target number of documents to select from each box. Defaults to 5.

        Returns:
            dict: An ECF dictionary containing the 'ExperimentName', the list of 
                  selected 'TrainingDocuments', and the 'Topics'.
        """
        random.seed(seed)
        topics = self.get_topics()
        
        if sampling == 'uniform':
            training_set = []
            training_files = []
            
            for box, folders in self.full_collection.items():
                folder_docs_limit = [0] * max_docs
                total_selected = 0
                
                # Determine how many docs per folder to pick
                # Logic mirrors original: distribute 5 docs across available folders
                num_folders = len(folders)
                for _ in range(docs_per_box):
                    for i in range(min(num_folders, docs_per_box, max_docs)):
                        if total_selected < docs_per_box:
                            folder_docs_limit[i] += 1
                            total_selected += 1
                
                folder_names = list(folders.keys())
                random.shuffle(folder_names)
                
                for i, folder in enumerate(folder_names):
                    candidates = [doc for doc in folders[folder] if doc not in training_files]
                    limit = folder_docs_limit[i] if i < len(folder_docs_limit) else 0
                    num_to_pick = min(limit, len(candidates))
                    
                    if num_to_pick > 0:
                        selected = random.sample(candidates, num_to_pick)
                        for doc in selected:
                            training_files.append(doc)
                            training_set.append(f"{box}/{folder}/{doc}")
            
            training_set.sort()
            
            ecf = {
                'ExperimentName': f'ECF w/ Random Seed {seed}',
                'ExperimentSets': [{'TrainingDocuments': training_set, 'Topics': {}}]
            }
            for topic in topics:
                ecf['ExperimentSets'][0]['Topics'][topic['ID']] = topic
            
            return ecf
        return None

    def load_all_docs_ecf(self):
        """
        Loads the 'Oracle' ECF file.

        This loads a specific, pre-generated ECF file that includes *every* document in the collection.
        """
        return self._load_json(self.all_docs_ecf_path)