import os
import json
from pathlib import Path
from kaggle_crawler import knowledge_base_generator

# the competition description and notebook solution are in json files
def process_kaggle_json(competition_dir, notebook_dir, knowledge_base_dir):
    # competition_dir = Path("/data/userdata/share/kaggle")
    # notebook_dir = Path("/data/userdata/share/kaggle/notebooks")
    # knowledge_base_dir = Path("/data/userdata/share/knowledge_base/knowledge_json")

    knowledge_base_dir.mkdir(parents=True, exist_ok=True)

    for competition_json in competition_dir.glob("*.json"):
        with open(competition_json, 'r', encoding='utf-8') as f:
            try:
                competition_data = json.load(f)
            except json.JSONDecodeError:
                output_filename = knowledge_base_dir / f"{competition_json.stem}_error.json"
                with open(output_filename, 'w', encoding='utf-8') as error_file:
                    json.dump({"Error": "Failed to parse competition JSON file."}, error_file, indent=4)
                continue
        
        competition_name = competition_json.stem
        competition_notebook_folder = notebook_dir / competition_name
        
        if not competition_notebook_folder.exists():
            output_filename = knowledge_base_dir / f"{competition_name}_error.json"
            with open(output_filename, 'w', encoding='utf-8') as error_file:
                json.dump({"Error": f"Notebook folder for competition '{competition_name}' not found."}, error_file, indent=4)
            continue
        
        for solution_folder in competition_notebook_folder.iterdir():
            if solution_folder.is_dir():
                solution_text_file = next(solution_folder.glob("*.txt"), None)
                
                if solution_text_file:
                    with open(solution_text_file, 'r', encoding='utf-8') as solution_file:
                        solution_text = solution_file.read()
                    
                    try:
                        response_dict = knowledge_base_generator(competition_text=competition_data, notebook_text=solution_text)
                    except Exception as e:
                        output_filename = knowledge_base_dir / f"{competition_name}_{solution_folder.name}_error.json"
                        with open(output_filename, 'w', encoding='utf-8') as error_file:
                            json.dump({"Error": f"Error during knowledge generation: {str(e)}"}, error_file, indent=4)
                        continue

                    if response_dict:
                        output_filename = knowledge_base_dir / f"{competition_name}_{solution_folder.name}.json"
                        with open(output_filename, 'w', encoding='utf-8') as output_file:
                            json.dump(response_dict, output_file, indent=4)
                        print(f"Generated knowledge base for {competition_name} - {solution_folder.name}")
                    else:
                        output_filename = knowledge_base_dir / f"{competition_name}_{solution_folder.name}_error.json"
                        with open(output_filename, 'w', encoding='utf-8') as error_file:
                            json.dump({"Error": "Knowledge base generator returned an empty response."}, error_file, indent=4)
                else:
                    output_filename = knowledge_base_dir / f"{competition_name}_{solution_folder.name}_error.json"
                    with open(output_filename, 'w', encoding='utf-8') as error_file:
                        json.dump({"Error": f"No .txt file found in solution folder '{solution_folder.name}'."}, error_file, indent=4)
            else:
                output_filename = knowledge_base_dir / f"{competition_name}_error.json"
                with open(output_filename, 'w', encoding='utf-8') as error_file:
                    json.dump({"Error": f"Skipping non-directory item: {solution_folder}."}, error_file, indent=4)
                
    print("Processing complete.")

# the competition description and notebook solution are in markdown files
def process_kaggle_md(competition_dir, notebook_dir, knowledge_base_dir):
    # competition_dir = Path("/data/userdata/share/mle_kaggle")
    # notebook_dir = Path("/data/userdata/share/kaggle/notebooks")
    # knowledge_base_dir = Path("/data/userdata/share/knowledge_base/knowledge_json2")

    knowledge_base_dir.mkdir(parents=True, exist_ok=True)

    for competition_folder in competition_dir.iterdir():
        if competition_folder.is_dir():
            competition_name = competition_folder.stem 
            competition_notebook_folder = notebook_dir / competition_name

            for competition_md in competition_folder.glob("*.md"):
                try:
                    with open(competition_md, 'r', encoding='utf-8') as f:
                        competition_data = f.read()
                except Exception:
                    continue

                if not competition_notebook_folder.exists():
                    continue

                for solution_folder in competition_notebook_folder.iterdir():
                    if solution_folder.is_dir():
                        solution_md_file = next(solution_folder.glob("*.md"), None)

                        if solution_md_file:
                            try:
                                with open(solution_md_file, 'r', encoding='utf-8') as solution_file:
                                    solution_text = solution_file.read()

                                response_dict = knowledge_base_generator(competition_text=competition_data, notebook_text=solution_text)
                                if response_dict:
                                    output_filename = knowledge_base_dir / f"{competition_name}_{solution_folder.name}.json"
                                    with open(output_filename, 'w', encoding='utf-8') as output_file:
                                        json.dump(response_dict, output_file, indent=4)
                            except Exception:
                                continue

    print("Processing complete.")

# the competition description is in json file and notebook solution is in markdown files
def process_kaggle(competition_dir, notebook_dir, knowledge_base_dir):
    # competition_dir = Path("/data/userdata/share/kaggle")
    # notebook_dir = Path("/data/userdata/share/kaggle/notebooks")
    # knowledge_base_dir = Path("/data/userdata/share/knowledge_base/knowledge_json2")

    knowledge_base_dir.mkdir(parents=True, exist_ok=True)

    for competition_json in competition_dir.glob("*.json"):
        with open(competition_json, 'r', encoding='utf-8') as f:
            try:
                competition_data = json.load(f)
            except json.JSONDecodeError:
                output_filename = knowledge_base_dir / f"{competition_json.stem}_error.json"
                with open(output_filename, 'w', encoding='utf-8') as error_file:
                    json.dump({"Error": "Failed to parse competition JSON file."}, error_file, indent=4)
                continue
        
        competition_name = competition_json.stem
        competition_notebook_folder = notebook_dir / competition_name

        if not competition_notebook_folder.exists():
            continue

        for solution_folder in competition_notebook_folder.iterdir():
            if solution_folder.is_dir():
                solution_md_file = next(solution_folder.glob("*.md"), None)

                if solution_md_file:
                    with open(solution_md_file, 'r', encoding='utf-8') as solution_file:
                        solution_text = solution_file.read()

                    response_dict = knowledge_base_generator(competition_text=competition_data, notebook_text=solution_text)
                    
                    if response_dict:
                        output_filename = knowledge_base_dir / f"{competition_name}_{solution_folder.name}.json"
                        with open(output_filename, 'w', encoding='utf-8') as output_file:
                            json.dump(response_dict, output_file, indent=4)

    print("Processing complete.")


competition_dir = Path("/data/userdata/share/kaggle")
notebook_dir = Path("/data/userdata/share/kaggle/notebooks")
knowledge_base_dir = Path("scripts/exp/researcher/output_ideas")
process_kaggle(competition_dir, notebook_dir, knowledge_base_dir)

