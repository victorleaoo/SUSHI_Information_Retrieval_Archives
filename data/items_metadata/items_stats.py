import json
import statistics

def analyze_ocr_pages(json_filepath):
    # Load the JSON data
    with open(json_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    page_counts = []

    # Iterate through each document in the JSON
    for doc_id, doc_data in data.items():
        # Check if 'ocr' exists and is a list to avoid errors
        if 'ocr' in doc_data and isinstance(doc_data['ocr'], list):
            num_pages = len(doc_data['ocr'])
            page_counts.append(num_pages)
        else:
            # Handle cases where 'ocr' might be missing or not a list (optional)
            page_counts.append(0)

    # If the collection is empty, handle it gracefully
    if not page_counts:
        print("No OCR data found to analyze.")
        return

    # Compute statistics
    min_pages = min(page_counts)
    max_pages = max(page_counts)
    mean_pages = statistics.mean(page_counts)
    median_pages = statistics.median(page_counts)
    
    # Standard deviation requires at least 2 data points
    if len(page_counts) > 1:
        std_dev_pages = statistics.stdev(page_counts)
    else:
        std_dev_pages = 0.0

    # Print the results
    print(f"Total Documents Analyzed: {len(page_counts)}")
    print("-" * 30)
    print(f"Minimum pages:      {min_pages}")
    print(f"Maximum pages:      {max_pages}")
    print(f"Mean (Average):     {mean_pages:.2f}")
    print(f"Median:             {median_pages:.2f}")
    print(f"Standard Deviation: {std_dev_pages:.2f}")

# Example usage:
# Save your JSON data in a file named 'documents.json' in the same folder as this script
if __name__ == "__main__":
    analyze_ocr_pages('itemsV1.2.json')