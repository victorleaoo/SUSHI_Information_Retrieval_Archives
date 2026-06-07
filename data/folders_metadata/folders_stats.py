import json

def analyze_snc_scopes(json_filepath):
    # Load the JSON data
    with open(json_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Dictionary to keep track of distinct SNCs
    # Format: { "SNC_NAME": boolean_has_scope }
    distinct_sncs = {}

    for folder_id, metadata in data.items():
        snc = metadata.get("snc")
        
        # Skip if the item doesn't have an SNC
        if not snc or snc.lower() == "nan":
            continue

        # Check if scope_truncated exists and is not empty
        scope = metadata.get("scope_truncated", "").strip()
        has_scope = bool(scope and scope.lower() != "nan")

        # Add to our tracker
        if snc not in distinct_sncs:
            distinct_sncs[snc] = has_scope
        elif has_scope:
            # If we've seen this SNC before but it didn't have a scope then, 
            # and now we found a scope for it, update it to True.
            distinct_sncs[snc] = True

    # Calculate the final statistics
    total_distinct = len(distinct_sncs)
    with_scope = sum(distinct_sncs.values())
    without_scope = total_distinct - with_scope

    # Print the results
    print("=== SNC Analysis ===")
    print(f"Total distinct SNCs: {total_distinct}")
    print(f"SNCs WITH a scope:   {with_scope}")
    print(f"SNCs WITHOUT a scope:{without_scope}")
    
    # Optional: If you want to see exactly which ones are missing a scope
    # missing_scopes = [snc for snc, has_scope in distinct_sncs.items() if not has_scope]
    # print(f"\nExample SNCs missing scope: {missing_scopes[:5]}")

# --- Usage Example ---
# Save your json as 'folders.json' and run the script
if __name__ == "__main__":
    analyze_snc_scopes('FoldersV1.3.json')