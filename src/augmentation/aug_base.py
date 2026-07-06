import json
import math
import random
from collections import Counter
from utils.llm_client import LLMClient

def _bm25_score(query, doc_text, avgdl, N, df_map, k1=1.5, b=0.75):
    q_terms = [t for t in query.lower().split() if len(t) > 2]
    d_terms = [t for t in doc_text.lower().split() if len(t) > 2]
    d_len = len(d_terms)
    if d_len == 0: return 0
    d_cnt = Counter(d_terms)
    score = 0
    for q in q_terms:
        if q not in d_cnt: continue
        idf = math.log((N - df_map.get(q, 0) + 0.5) / (df_map.get(q, 0) + 0.5) + 1)
        tf = d_cnt[q]
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * d_len / (avgdl + 1e-9)))
    return score

def select_docs_for_folder(folder_label, doc_ids, all_docs_metadata, max_docs=5):
    """
    Selects documents based on the strategy:
    - If <= 5: use all
    - If > 5: 2 random, top 3 by BM25 against folder label
    """
    if len(doc_ids) <= max_docs:
        return doc_ids

    # Separate logic for > 5
    random.seed(42)  # For deterministic selection
    sorted_doc_ids = sorted(doc_ids) # keep it stable
    random_selection = random.sample(sorted_doc_ids, 2)
    
    remaining_docs = [did for did in sorted_doc_ids if did not in random_selection]
    
    # Calculate simple BM25
    doc_texts = {did: f"{all_docs_metadata[did].get('title', '')} {all_docs_metadata[did].get('summary', '')}" for did in remaining_docs}
    N = len(remaining_docs)
    d_terms_all = {did: [t for t in doc_texts[did].lower().split() if len(t) > 2] for did in remaining_docs}
    df_map = Counter()
    total_len = 0
    for terms in d_terms_all.values():
        total_len += len(terms)
        for t in set(terms):
            df_map[t] += 1
    
    avgdl = total_len / N if N > 0 else 1
    
    scores = []
    for did in remaining_docs:
        score = _bm25_score(folder_label, doc_texts[did], avgdl, N, df_map)
        scores.append((score, did))
        
    scores.sort(reverse=True, key=lambda x: x[0])
    top_bm25 = [did for _, did in scores[:3]]
    
    final_selection = random_selection + top_bm25
    return final_selection

def extract_description_keywords(llm_output):
    """Parses DESCRIPTION and KEYWORDS from LLM response"""
    lines = llm_output.split('\n')
    desc = ""
    keywords = ""
    current_section = None
    for line in lines:
        if line.startswith("DESCRIPTION:"):
            current_section = "desc"
            desc += line.replace("DESCRIPTION:", "").strip() + " "
        elif line.startswith("KEYWORDS:"):
            current_section = "keywords"
            keywords += line.replace("KEYWORDS:", "").strip() + " "
        elif current_section == "desc" and not line.startswith("KEYWORDS:"):
            desc += line.strip() + " "
        elif current_section == "keywords":
            keywords += line.strip() + " "
            
    return desc.strip(), keywords.strip()

def augment_with_docs(folder_meta, selected_docs, all_docs_metadata, llm_client: LLMClient):
    folder_label = folder_meta.get('title', '')
    snc = folder_meta.get('snc', 'Unknown')
    if not snc or snc.strip() == '': snc = "Unknown"
    
    snc_code = "Unclassified" if snc == "Unknown" else snc
    expanded_snc = folder_meta.get('expanded_snc', folder_label) if snc == "Unknown" else folder_meta.get('expanded_snc', '')
    
    parent_expanded = folder_meta.get('label_parent_expanded', '')
    if not parent_expanded or parent_expanded == snc:
        parent_expanded = folder_meta.get('main_title', '')
        
    scope_note = folder_meta.get('scope_note', '')
    if not scope_note: scope_note = "No scope note available for this SNC code."
    
    start_date = folder_meta.get('start_date', 'Unknown')
    end_date = folder_meta.get('end_date', 'Unknown')
    date_range = f"{start_date} to {end_date}" if end_date != "Unknown" else f"{start_date} (end date not recorded)"
    
    doc_list_str = ""
    for idx, did in enumerate(selected_docs):
        doc_meta = all_docs_metadata.get(did, {})
        title = doc_meta.get('title', '')
        summary = doc_meta.get('summary', '')
        doc_list_str += f"[Doc {idx+1}] Title: {title} | Summary: {summary}\n"
        
    prompt = f"""You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for the archival folder below.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Code: {snc_code}
- SNC Meaning: {expanded_snc}
- Broader Category: {parent_expanded}
- Scope Note: {scope_note}
- Date Range: {date_range}

DOCUMENTS IN THIS FOLDER:
{doc_list_str}

Think through these steps before writing:
Step 1 — ENTITIES: List the specific people, organizations, institutions, and locations mentioned across the documents.
Step 2 — EVENTS: List the specific events, actions, decisions, or developments described.
Step 3 — THEMES: Identify 2-3 overarching themes that connect the documents.

Then output the following two sections:

DESCRIPTION: A paragraph of 100-150 words starting exactly with "This folder contains" describing the specific historical content, key actors, events, and context in these documents. Be specific — name individuals, institutions, and events. Do not use generic phrases like "various documents" or "multiple topics."

KEYWORDS: A comma-separated list of 10-15 search terms: proper nouns, event names, institutions, and key concepts that best represent this folder's content.

Output only the DESCRIPTION and KEYWORDS sections. Do not output your step-by-step reasoning."""

    response = llm_client.generate(prompt)
    desc, kw = extract_description_keywords(response)
    # the index wants: {original_folder_label} | {expanded_snc} | {scope_note} | {DESCRIPTION} | {KEYWORDS}
    sn_part = scope_note if scope_note != "No scope note available for this SNC code." else ""
    # We will format this outside, just return desc and keywords.
    return desc, kw, prompt, response

def augment_no_docs(folder_meta, llm_client: LLMClient):
    folder_label = folder_meta.get('title', '')
    snc = folder_meta.get('snc', 'Unknown')
    if not snc or snc.strip() == '': snc = "Unknown"
    
    snc_code = "Unclassified" if snc == "Unknown" else snc
    expanded_snc = folder_meta.get('expanded_snc', folder_label) if snc == "Unknown" else folder_meta.get('expanded_snc', '')
    
    parent_expanded = folder_meta.get('label_parent_expanded', '')
    if not parent_expanded or parent_expanded == snc:
        parent_expanded = folder_meta.get('main_title', '')
        
    scope_note = folder_meta.get('scope_note', '')
    if not scope_note: scope_note = "No scope note available for this SNC code."
    
    start_date = folder_meta.get('start_date', 'Unknown')
    end_date = folder_meta.get('end_date', 'Unknown')
    date_range = f"{start_date} to {end_date}" if end_date != "Unknown" else f"{start_date} (end date not recorded)"
    
    context_date = f"The period covers {start_date}–{end_date}." if end_date != "Unknown" else "The end date is not recorded, but based on the classification and folder context, content likely extends through the late 1960s or early 1970s."
    
    prompt = f"""You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for an archival folder that has not yet been digitized.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Code: {snc_code}
- SNC Meaning: {expanded_snc}
- Broader Category: {parent_expanded}
- Scope Note: {scope_note}
- Date Range: {date_range}
- Record Group: {folder_meta.get('record_group', '')}

HISTORICAL CONTEXT:
This folder is from U.S. State Department records on Brazil, starting from {start_date}.
{context_date}
The broader historical period includes: the April 1964 military coup and subsequent military government, Cold War dynamics in Latin America, significant Brazilian economic development, and evolving U.S.-Brazil bilateral relations across security, trade, and diplomacy.

Think through these steps before writing:
Step 1 — SNC INTERPRETATION: What specific type of content does this SNC code cover? What would a State Department officer file under this label?
Step 2 — HISTORICAL EVENTS: What specific events, actors, and developments occurred in Brazil and in U.S.-Brazil relations relevant to this topic during the folder's time period?
Step 3 — DOCUMENT INFERENCE: What types of documents (cables, memos, intelligence reports, diplomatic notes) would likely populate this folder?

Then output the following two sections:

DESCRIPTION: A paragraph of 100-150 words starting exactly with "This folder likely contains" describing the probable historical content, likely key actors and institutions, and relevant events. Name known historical figures, institutions, and events from this era. Do not invent document contents — draw only from established historical knowledge.

KEYWORDS: A comma-separated list of 10-15 search terms including proper nouns, event names, institutions, and key concepts most likely to represent this folder.

Output only the DESCRIPTION and KEYWORDS sections. Do not output your step-by-step reasoning."""

    response = llm_client.generate(prompt)
    desc, kw = extract_description_keywords(response)
    return desc, kw, prompt, response
