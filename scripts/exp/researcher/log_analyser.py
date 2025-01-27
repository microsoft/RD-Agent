import os
import re
from collections import Counter
import matplotlib.pyplot as plt

def preprocess_ideas(ideas):
    """
    Preprocess the ideas to merge those containing specific keywords into unified categories.

    Args:
        ideas (list): A list of idea strings.

    Returns:
        list: A processed list of ideas with unified categories.
    """
    unified_ideas = []
    for idea in ideas:
        if "ensemble" in idea.lower() or "ensembling" in idea.lower():
            unified_ideas.append("Model Ensembling")
        elif "feature engineering" in idea.lower():
            unified_ideas.append("Feature Engineering")
        elif "data augmentation" in idea.lower():
            unified_ideas.append("Data Augmentation")
        elif "stacking" in idea.lower():
            unified_ideas.append("Model Stacking")
        elif "selection" in idea.lower():
            unified_ideas.append("Feature Selection")
        else:
            unified_ideas.append(idea)  # Keep other ideas unchanged
    return unified_ideas


def extract_ideas_from_logs(base_dir):
    """
    Extract all "idea" values from log files located within a specific directory structure,
    keeping only one occurrence of each idea from every triplet.

    Args:
        base_dir (str): The base directory to start the search.

    Returns:
        list: A list of all extracted "idea" values with duplicates removed within triplets.
    """
    ideas = []  # List to store all extracted "idea" values

    # Walk through the base directory
    for root, dirs, _ in os.walk(base_dir):
        if "llm_messages" in dirs:
            llm_messages_path = os.path.join(root, "llm_messages")
            for subdir in os.listdir(llm_messages_path):
                subfolder_path = os.path.join(llm_messages_path, subdir)
                log_file_path = os.path.join(subfolder_path, "common_logs.log")
                if os.path.isfile(log_file_path):
                    with open(log_file_path, "r", encoding="utf-8") as log_file:
                        for line in log_file:
                            match = re.search(r'"idea"\s*:\s*"([^"]+)"', line)
                            if match:
                                ideas.append(match.group(1))
   
    # Filter out duplicates within triplets
    filtered_ideas = []
    seen_counts = Counter()
    for idea in ideas:
        if seen_counts[idea] % 3 == 0:  # Add only one occurrence for every triplet
            filtered_ideas.append(idea)
        seen_counts[idea] += 1

    # Preprocess ideas to merge specific categories
    return preprocess_ideas(filtered_ideas)


def visualize_top_10_ideas(ideas, output_path):
    """
    Visualize the top 10 most frequent ideas in a list and save the plot as an image.

    Args:
        ideas (list): A list of idea names.
        output_path (str): Path to save the visualization.

    Returns:
        None
    """
    # Count the frequency of each idea
    idea_counts = Counter(ideas)
   
    # Sort ideas by frequency in descending order and keep only the top 10
    sorted_ideas = sorted(idea_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    idea_names, frequencies = zip(*sorted_ideas)
   
    # Print the top 10 ideas
    print("Top 10 Ideas by Frequency:")
    for i, (idea, count) in enumerate(sorted_ideas, start=1):
        print(f"{i}. {idea}: {count} occurrences")
   
    # Plot the top 10 ideas
    plt.figure(figsize=(10, 6))
    plt.bar(idea_names, frequencies, color='skyblue')
    plt.xlabel("Idea Names")
    plt.ylabel("Frequency")
    plt.title("Top 10 Ideas by Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
   
    # Save the plot to the specified path
    plt.savefig(output_path)
    plt.show()


# Main function
if __name__ == "__main__":
    # Define the base directory
    base_directory = "/data/userdata/share/researcher/log"
   
    # Extract ideas from logs
    extracted_ideas = extract_ideas_from_logs(base_directory)
   
    # Define the output path for the visualization
    output_file = os.path.join("/data/userdata/share/knowledge_base", "idea_frequencies.png")
   
    # Visualize and save the top 10 frequency of ideas
    visualize_top_10_ideas(extracted_ideas, output_file)
   
    print(f"Visualization saved to {output_file}")