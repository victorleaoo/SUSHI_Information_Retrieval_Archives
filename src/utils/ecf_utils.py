import json

def load_ecf(ecf_path: str) -> dict:
    """Loads an ECF JSON file."""
    with open(ecf_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_training_docs_per_folder(ecf: dict, items: dict, folders: dict = None) -> dict:
    """Maps each folder to its training document IDs from the ECF."""
    folder_docs = {}  # {folder_id: [doc_id, ...]}
    if "ExperimentSets" in ecf and len(ecf["ExperimentSets"]) > 0:
        for doc_path in ecf['ExperimentSets'][0].get('TrainingDocuments', []):
            doc_id = doc_path[-10:-4]  # Extract doc ID from path
            if doc_id in items:
                folder_id = items[doc_id]['Sushi Folder']
                folder_docs.setdefault(folder_id, []).append(doc_id)
    return folder_docs
