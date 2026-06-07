import json
import statistics

def analyze_topic_word_counts(json_filepath):
    # Load the JSON data
    with open(json_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    topics = data["ExperimentSets"][0]["Topics"]
    
    # Lists to store the word counts for each field
    title_word_counts = []
    description_word_counts = []
    narrative_word_counts = []

    for topic_id, topic_data in topics.items():
        # Get the text for each field (defaulting to empty string if missing)
        title_text = topic_data.get("TITLE", "")
        desc_text = topic_data.get("DESCRIPTION", "")
        narr_text = topic_data.get("NARRATIVE", "")

        # Split the text by whitespace to count words and append to our lists
        title_word_counts.append(len(title_text.split()))
        description_word_counts.append(len(desc_text.split()))
        narrative_word_counts.append(len(narr_text.split()))

    # If there are no topics, handle it gracefully
    if not title_word_counts:
        print("No topics found to analyze.")
        return

    # Calculate statistics
    avg_title = statistics.mean(title_word_counts)
    avg_desc = statistics.mean(description_word_counts)
    avg_narr = statistics.mean(narrative_word_counts)

    # Print the results
    print(f"=== Topic Word Count Analysis ({len(topics)} topics) ===")
    print(f"Average words in TITLE:       {avg_title:.2f}")
    print(f"Average words in DESCRIPTION: {avg_desc:.2f}")
    print(f"Average words in NARRATIVE:   {avg_narr:.2f}")
    
    # Optional: You can also print Min and Max for extra insight
    print("-" * 40)
    print(f"TITLE Range:       {min(title_word_counts)} to {max(title_word_counts)} words")
    print(f"DESCRIPTION Range: {min(description_word_counts)} to {max(description_word_counts)} words")
    print(f"NARRATIVE Range:   {min(narrative_word_counts)} to {max(narrative_word_counts)} words")

# --- Usage Example ---
# Save your json as 'topics.json' and run the script
if __name__ == "__main__":
    analyze_topic_word_counts('./random_generated/ECF_RANDOM_1.json')