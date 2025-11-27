import json
import os

def extract_topics(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    with open(output_file, 'w') as out:
        if "ExperimentSets" in data:
            for experiment_set in data["ExperimentSets"]:
                if "Topics" in experiment_set:
                    topics = experiment_set["Topics"]
                    for topic_id, topic_data in topics.items():
                        out.write(f"# {topic_id}\n\n")
                        out.write("```json\n")
                        out.write(json.dumps(topic_data, indent=4))
                        out.write("\n```\n\n")
        else:
            print("Key 'ExperimentSets' not found in JSON.")

if __name__ == "__main__":
    # Adjust the filename if your input file has a different name or extension
    input_json = "Ntcir18SushiOfficialExperimentControlFileV1.1.json" 
    output_txt = "topics_output.txt"
    
    extract_topics(input_json, output_txt)
    print(f"Extraction complete. Check {output_txt}")