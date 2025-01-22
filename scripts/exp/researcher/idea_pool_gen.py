import os
import json
from pathlib import Path
from kaggle_crawler import knowledge_base_generator
import nbformat


def convert_notebooks_to_text(notebook_path):
    """Convert a Jupyter notebook to a text file."""
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    text = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            text.append(f"```markdown\n{cell.source}```")
        elif cell.cell_type == "code":
            text.append(f"```code\n{cell.source}```")
    
    return "\n\n".join(text)

def process_competitions(base_path, output_path):
    """
    Process competition folders to extract ideas from competition descriptions and solutions.

    Parameters:
        base_path (str): Path to the directory containing competition folders.
        output_path (str): Path to the directory where output JSON files will be saved.
    """
    base_dir = Path(base_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ideas = []

    for competition_folder in base_dir.iterdir():
        if not competition_folder.is_dir():
            continue

        competition_name = competition_folder.name
        json_file = competition_folder / f"{competition_name}.json"
        solutions_dir = competition_folder / competition_name

        if not json_file.exists() or not solutions_dir.is_dir():
            continue

        # Read competition description
        with open(json_file, "r", encoding="utf-8") as f:
            competition_data = json.load(f)

        # Process each solution
        for solution_folder in solutions_dir.iterdir():
            if not solution_folder.is_dir():
                continue

            solution_text = ""

            for notebook_path in solution_folder.glob("*.ipynb"):
                text = convert_notebooks_to_text(notebook_path)
                solution_text += text + "\n\n"

            # Save combined solution text as .txt
            combined_text_path = solution_folder / "solution.txt"
            with combined_text_path.open("w", encoding="utf-8") as f:
                f.write(solution_text)

            # Generate ideas
            ideas_dict = knowledge_base_generator(
                competition_text=competition_data, 
                notebook_text=solution_text
            )

            # Save ideas to JSON file
            output_file = output_dir / f"{competition_name}_{solution_folder.name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(ideas_dict, f, indent=4)

            all_ideas.append(ideas_dict)

    # Save all ideas to a single JSON file
    all_ideas_file = output_dir / "all_ideas.json"
    with open(all_ideas_file, "w", encoding="utf-8") as f:
        json.dump(all_ideas, f, indent=4)

# Example usage
if __name__ == "__main__":
    base_path = "/data/userdata/v-xhong/ds_researcher/RD-Agent/scripts/exp/researcher/training_set/raw_jsons"
    output_path = "/data/userdata/v-xhong/ds_researcher/RD-Agent/scripts/exp/researcher/training_set/extracted_ideas"
    process_competitions(base_path, output_path)