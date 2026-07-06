from utils.llm_client import LLMClient

def generate_tcde_label(folder_meta, selected_docs, all_docs_metadata, llm_client: LLMClient):
    folder_label = folder_meta.get('title', '')
    snc = folder_meta.get('snc', 'Unknown')
    snc_code = "Unclassified" if snc == "Unknown" or not snc.strip() else snc
    expanded_snc = folder_meta.get('expanded_snc', folder_label) if snc_code == "Unclassified" else folder_meta.get('expanded_snc', '')
    
    start_date = folder_meta.get('start_date', 'Unknown')
    end_date = folder_meta.get('end_date', 'Unknown')
    
    # Pass 1: Identify Topics
    has_docs = selected_docs is not None and len(selected_docs) > 0
    
    if has_docs:
        doc_list_str = ""
        for idx, did in enumerate(selected_docs):
            doc_meta = all_docs_metadata.get(did, {})
            title = doc_meta.get('title', '')
            summary = doc_meta.get('summary', '')
            doc_list_str += f"[Doc {idx+1}] Title: {title} | Summary: {summary}\n"
        doc_context = f"DOCUMENT CONTENT (sample):\n{doc_list_str}"
    else:
        doc_context = f"No documents are digitized for this folder. Base your analysis on the folder label, SNC classification, and knowledge of the {start_date}–{end_date} period."
        
    prompt_topic = f"""You are an expert archivist specializing in U.S. State Department records on Brazil (1960s-1970s). Identify exactly 5 abstract topics for the archival folder below.

FOLDER INFORMATION:
- Label: {folder_label}
- SNC: {expanded_snc}
- Date Range: {start_date} to {end_date}

{doc_context}

Identify exactly 5 DISTINCT topics. Each topic must:
- Be expressed as one specific sentence (not a general category)
- Cover a different aspect of the folder's likely content
- Reference specific types of actors, events, or issues (not generic phrases)
- Be grounded in the {start_date}–{end_date} historical context

Return exactly 5 topic sentences. One per line. No numbering, no preamble."""

    response_topics_raw = llm_client.generate(prompt_topic)
    response_topics = response_topics_raw.strip().split('\n')
    topics = [t.strip() for t in response_topics if t.strip()]
    
    # Ensure exactly 5 topics
    if len(topics) < 5:
        topics += [f"Aspect {i+1} of {folder_label}." for i in range(5 - len(topics))]
    topics = topics[:5]
    
    # Formatting for Pass 2
    topics_str = ""
    for idx, t in enumerate(topics):
        topics_str += f"{{topic_{idx+1}}}\n{t}\n"
        
    # Pass 2: Expand Topics
    prompt_expand = f"""You are a diplomatic historian specializing in U.S.-Brazil relations (1960s-1970s). Below are 5 topics for an archival folder ({folder_label}, {start_date}–{end_date}).

TOPICS:
{topics_str}

For each topic, write one detailed expansion sentence that:
- Names specific historical actors, institutions, or events related to that topic
- Uses terminology consistent with U.S. State Department language from the 1960s-1970s
- Provides concrete historical context useful for a researcher searching these archives

Return exactly 5 sentences, one per line, in the same order as the topics. No numbering, no preamble."""

    response_expansions_raw = llm_client.generate(prompt_expand)
    response_expansions = response_expansions_raw.strip().split('\n')
    expansions = [e.strip() for e in response_expansions if e.strip()]
    
    if len(expansions) < 5:
        expansions += [f"Detailed aspect {i+1}." for i in range(5 - len(expansions))]
    expansions = expansions[:5]
    
    # Combine
    combined = " ".join(topics) + " " + " ".join(expansions)
    
    prompts = {"prompt_topic": prompt_topic, "prompt_expand": prompt_expand}
    responses = {"response_topics": response_topics_raw, "response_expansions": response_expansions_raw}
    
    return combined, "", prompts, responses
