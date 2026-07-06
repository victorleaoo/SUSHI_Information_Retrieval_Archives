def resolve_folder_fields(folder: dict) -> dict:
    """Resolves all prompt variables from a folder metadata dict."""
    snc = folder.get('snc', 'Unknown')
    label = folder.get('label', '')  # ALWAYS populated
    main_title = folder.get('main_title', '')  # ALWAYS populated
    label_parent_expanded = folder.get('label_parent_expanded', '')
    
    # SNC display
    snc_display = 'Unclassified' if snc == 'Unknown' else snc
    
    # Expanded SNC: prefer label_parent_expanded, fall back to main_title, then label
    if label_parent_expanded and str(label_parent_expanded) != 'nan' and label_parent_expanded != snc:
        expanded_snc = label_parent_expanded
    elif main_title and str(main_title) != 'nan':
        expanded_snc = main_title
    else:
        expanded_snc = label  # Ultimate fallback — always populated
    
    # Parent expanded: broader category
    if label_parent_expanded and str(label_parent_expanded) != 'nan' and ':' in label_parent_expanded:
        parent_expanded = label_parent_expanded.split(':')[0].strip()
    elif main_title and str(main_title) != 'nan':
        parent_expanded = main_title
    else:
        parent_expanded = label
    
    # Scope note
    scope = folder.get('scope_truncated', '') or folder.get('raw_scope', '')
    if str(scope) == 'nan':
        scope = ''
    scope_display = scope if scope else 'No scope note available for this SNC code.'
    
    # Date range
    start_date = folder.get('date', '')
    end_date = folder.get('endDate', 'Unknown')
    if end_date == 'Unknown':
        date_range = f"{start_date} (end date not recorded)"
    else:
        date_range = f"{start_date} to {end_date}"
    
    # SNC hierarchy for cascading homophily
    parts = snc.split('-')[0].split()  # "POL 18-2" → ["POL", "18"]
    snc_2level = ' '.join(parts[:2]) if len(parts) >= 2 else snc
    parent = folder.get('parent', 'None')
    
    return {
        'folder_label': label,
        'snc': snc_display,
        'expanded_snc': expanded_snc,
        'parent_expanded_snc': parent_expanded,
        'scope_note': scope_display,
        'start_date': start_date,
        'end_date': end_date,
        'date_range_formatted': date_range,
        'record_group': folder.get('rg', ''),
        'main_title': main_title,
        'snc_2level': snc_2level,
        'parent': parent,
    }
