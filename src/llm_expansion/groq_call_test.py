from groq import Groq

client = Groq(api_key="")

system_content = """
The archive contains diplomatic cables, intelligence memos, and government files. However, many folders only have a generic SNC label and no documents. 

Your goal is to contextualize the generic SNC code specifically to the political, social, and military reality of Brazil between 1960 and 1985. 
"""

snc_code = "LAB 3"
snc_description = "LABOR & MANPOWER: ORGANIZATIONS & CONFERENCES"
snc_scope_note = ""

snc_content = f"""
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

chat_completion = client.chat.completions.create(
    messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": system_content,
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": snc_content,
        }
    ],

    # The language model which will generate the completion.
    model="qwen/qwen3-32b"
)

# Print the completion returned by the LLM.
print(chat_completion.choices[0].message.content)