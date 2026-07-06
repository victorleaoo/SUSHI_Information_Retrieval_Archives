def build_folder_index_text(folder: dict, config: str) -> str:
    """Builds the indexable text for a folder based on the F-config."""
    label = folder.get('label', '')
    main_title = folder.get('main_title', '')
    lpe = folder.get('label_parent_expanded', '')
    scope = folder.get('scope_truncated', '') or folder.get('raw_scope', '')
    
    # Resolve parent_expanded: use main_title as fallback
    parent_expanded = lpe if (lpe and str(lpe) != 'nan' and lpe != folder.get('snc')) else main_title
    
    if config == 'F1':
        return label
    elif config == 'F2':
        return f"{label} | {parent_expanded}"
    elif config == 'F3':
        if scope and str(scope) != 'nan':
            return f"{label} | {scope}"
        return label  # F3 degrades to F1 when no scope note
    elif config == 'F4':
        parts = [label, parent_expanded]
        if scope and str(scope) != 'nan':
            parts.append(str(scope))
        return ' | '.join(parts)
    elif config == 'F5':
        parts = [label, parent_expanded]
        if scope and str(scope) != 'nan':
            parts.append(str(scope))
        if main_title and str(main_title) != 'nan':
             parts.append(str(main_title))
        return ' | '.join(parts)
    
    return label

def build_allf_training_data(folders: dict, config: str) -> list:
    """Creates training data list in the format used by existing models.py"""
    training_data = []
    for folder_id, folder in folders.items():
        text = build_folder_index_text(folder, config)
        training_entry = {
            'docno': folder_id,       # Folder ID
            'folder': folder_id,      # Same as docno for folder-level indexing
            'box': folder.get('box', ''),
            'date': folder.get('date', ''),
            'folderlabel': text,  # F1-F5
            'text_blob': text,    # For dense models
        }
        training_data.append(training_entry)
    return training_data

def build_allf_index(folders: dict, config: str, model_name: str, models_module):
    """Instantiates the appropriate RetrievalModel subclass and calls .train(data)"""
    training_data = build_allf_training_data(folders, config)
    
    if model_name == 'bm25':
        # BM25Model takes current_searching_field in init
        model = models_module.BM25Model(['folderlabel']) 
    elif model_name == 'embeddings':
        model = models_module.EmbeddingsModel()
    elif model_name == 'colbert':
        model = models_module.ColBERTModel()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    model.train(training_data)
    return model
