import os
import json
from pathlib import Path
from kaggle_crawler import idea_merger

def merge_ideas(input_path1, input_path2, output_path, index1, index2):
    """
    Merge two idea JSON objects using the `idea_merger` function and save the result to an output file.

    Parameters:
    input_path1 (str): Path to the first input JSON file.
    input_path2 (str): Path to the second input JSON file.
    output_path (str): Path to save the merged output JSON file.
    index1 (int): Index of the idea to extract from the first JSON file. Default is 0.
    index2 (int): Index of the idea to extract from the second JSON file. Default is 4.

    Returns:
    None
    """
    try:
        # Load the first idea
        with open(input_path1, 'r', encoding='utf-8') as file1:
            idea1 = json.load(file1)
        s1 = json.dumps(idea1[index1])  # Extracting the specified index

        # Load the second idea
        with open(input_path2, 'r', encoding='utf-8') as file2:
            idea2 = json.load(file2)
        s2 = json.dumps(idea2[index2])  # Extracting the specified index

        # Merge ideas using `idea_merger`
        response_dict = idea_merger(old_idea=s1, new_idea=s2)

        # Save the merged result to the output file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(response_dict, output_file, indent=4)

        print(f"Merged ideas saved successfully to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# merge_ideas('/data/userdata/share/knowledge_base/knowledge_json2/playground-series-s3e24_alexryzhkov.json',
#             '/data/userdata/share/knowledge_base/knowledge_json2/playground-series-s3e24_cv13j0.json', 
#             '/data/userdata/share/knowledge_base/output_json.json', 0, 4)

