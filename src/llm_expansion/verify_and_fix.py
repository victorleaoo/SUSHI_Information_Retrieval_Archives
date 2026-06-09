import json
import time
import os
from groq import Groq

# Configuration
INPUT_FILE = "../../data/folders_metadata/expanded_snc.json"
API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

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

def verify_and_fix():
    file_path = os.path.join(os.path.dirname(__file__), INPUT_FILE)

    if not os.path.exists(file_path):
        print(f"File not found: {os.path.abspath(file_path)}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Total items in file: {len(data)}")
    
    issues = []
    
    # 1. Identify missing or problematic expansions
    for i, item in enumerate(data):
        expansion = item.get("expansion")
        
        # Check if expansion is properly structured
        if not isinstance(expansion, dict) or "error" in expansion:
            issues.append(i)
            continue
            
        # Check for empty dense_embedding_summary
        summary = expansion.get("dense_embedding_summary", "")
        if not summary or not str(summary).strip():
            issues.append(i)

    if not issues:
        print("Success! All 379 items have a valid and non-empty 'dense_embedding_summary'.")
        return

    print(f"Found {len(issues)} items with missing/empty summaries or errors. Fixing them now...")

    # 2. Fix the issues
    fixed_count = 0
    for idx in issues:
        item = data[idx]
        snc_code = item.get('snc', 'Unknown')
        print(f"Fixing SNC: {snc_code}...")
        
        prompt = generate_prompt(
            snc_code, 
            item.get('label_parent_extended', ''), 
            item.get('raw_scope', '')
        )
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_CONTENT},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL,
                temperature=0.2,
                max_completion_tokens=2000, # Increased to prevent "max completion tokens reached"
                response_format={"type": "json_object"}
            )
            content = chat_completion.choices[0].message.content
            
            if content:
                item['expansion'] = json.loads(content)
                fixed_count += 1
                print(f" -> Successfully generated replacement for {snc_code}")
            else:
                print(f" -> Warning: Empty response for {snc_code}")
                
            # Sleep 8 seconds to respect the 30 req/min limits
            time.sleep(8)
            
        except Exception as e:
            print(f" -> Failed to process {snc_code}: {e}")

    # 3. Save the fixed data back to the file
    if fixed_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {fixed_count} fixed objects to {file_path}")

    # 4. Final Verification
    missing_after_fix = []
    for i, item in enumerate(data):
        summary = item.get("expansion", {}).get("dense_embedding_summary", "")
        if not summary or not str(summary).strip():
            missing_after_fix.append(item.get("snc"))
            
    if missing_after_fix:
        print(f"Remaining missing summaries: {len(missing_after_fix)}")
    else:
        print("Final check passed: All items now have a complete dense embedding summary.")

if __name__ == "__main__":
    verify_and_fix()
