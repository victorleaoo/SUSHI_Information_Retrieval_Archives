import json
import time
import os
from groq import Groq

# Configuration
INPUT_FILE = "../../data/folders_metadata/distinct_snc.json"
OUTPUT_FILE = "../../data/folders_metadata/expanded_snc.json"
API_KEY = os.environ.get("GROQ_API_KEY", "") # Better to use env variable, fallback to hardcoded
MODEL = "llama-3.1-8b-instant"  # Using the model from selection

# Rate limiting
# token/min limit is 6k. We estimate ~750 tokens per request.
# 6000 / 750 = 8 requests per minute.
# 60 seconds / 8 requests = 7.5 seconds per request.
SLEEP_TIME_SECONDS = 8

client = Groq(api_key=API_KEY)

SYSTEM_CONTENT = """
The archive contains diplomatic cables, intelligence memos, and government files. However, many folders only have a generic SNC label and no documents. 

Your goal is to contextualize the generic SNC code specifically to the political, social, and military reality of Brazil between 1960 and 1985. 
"""

def generate_prompt(snc_code, snc_description, snc_scope_note):
    return f"""
INPUT DATA:
- SNC Code: {snc_code}
- SNC Description: {snc_description}
- Scope Note: {snc_scope_note}
- Sample Documents (if any): no sample documents

INSTRUCTIONS:
1. Analyze the generic SNC definition.
2. If "Sample Documents" are provided, extract specific terminology, entities, and events from them.
3. If "Sample Documents" is empty, use your historical knowledge to infer what specific Brazilian events, people, institutions, or regions from the 1960s-1980s would logically be filed under this generic code.
4. Generate a "Rich Folder Profile" optimized for both lexical (BM25) and dense (Vector) search.

Output your response STRICTLY as a JSON object with the following structure:

{{
  "enhanced_folder_title": "A descriptive, 10-15 word title bridging the SNC code with the Brazilian historical context.",
  "dense_embedding_summary": "A 2-paragraph synthetic summary describing the exact types of historical events, memos, and discussions regarding Brazil (1960-1985) that would be found in this folder. Write this in the formal, analytical tone of a U.S. diplomatic archivist. This will be used for vector embeddings.",
  "bm25_keywords": {{
    "historical_entities": ["Specific Brazilian political parties (e.g., ARENA, MDB), government branches, military units, or opposition groups relevant to this code"],
    "key_figures": ["Specific politicians, generals, activists, or U.S. diplomats relevant to this topic during the era"],
    "bureaucratic_jargon": ["U.S. diplomatic and intelligence terms related to this code"],
    "locations": ["Specific Brazilian states, cities, or regions if the code warrants geographic focus"]
  }}
}}

Output ONLY valid JSON. Do not include markdown blocks or introductory text.
"""

def expand_folders():
    input_path = os.path.join(os.path.dirname(__file__), INPUT_FILE)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)

    with open(input_path, 'r', encoding='utf-8') as f:
        snc_data = json.load(f)

    # Load existing progress if available
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            expanded_data = json.load(f)
        processed_sncs = {item['snc'] for item in expanded_data}
    else:
        expanded_data = []
        processed_sncs = set()

    for idx, item in enumerate(snc_data):
        snc_code = item.get('snc', '')
        if snc_code in processed_sncs:
            print(f"[{idx+1}/{len(snc_data)}] Skipping already processed SNC: {snc_code}")
            continue
            
        print(f"[{idx+1}/{len(snc_data)}] Processing SNC: {snc_code}...")

        snc_description = item.get('label_parent_extended', '')
        snc_scope_note = item.get('raw_scope', '')

        prompt = generate_prompt(snc_code, snc_description, snc_scope_note)

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_CONTENT},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL,
                temperature=0.2, # slightly low temperature for consistency
                response_format={"type": "json_object"}
            )

            # Ensure we have content
            content = chat_completion.choices[0].message.content
            if content:
                llm_response = json.loads(content)
                item['expansion'] = llm_response
            else:
                print(f"Warning: Empty response for SNC {snc_code}")
                item['expansion'] = {}

        except Exception as e:
            print(f"Error processing {snc_code}: {e}")
            item['expansion'] = {"error": str(e)}
        
        expanded_data.append(item)

        # Save progress after each iteration
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(expanded_data, f, indent=2, ensure_ascii=False)

        # Rate limiting sleep
        if idx < len(snc_data) - 1:
            print(f"Sleeping for {SLEEP_TIME_SECONDS} seconds to respect rate limits...")
            time.sleep(SLEEP_TIME_SECONDS)

if __name__ == "__main__":
    expand_folders()
