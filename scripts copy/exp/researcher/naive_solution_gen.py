import os
import json
from pathlib import Path
from kaggle_crawler import naive_solution_gen

def naive_solution_generator(competition_name, competition_dir):
    """
    Generate a naive solution based on the competition description file.

    Args:
        competition_name (str): Name of the competition.
        competition_dir (Path): Directory where competition files are stored.

    Returns:
        The result of the naive solution generation.
    """
    # Construct the path to the competition description file
    competition_file = competition_dir / f"{competition_name}.json"

    # Open and read the competition description
    with open(competition_file, 'r', encoding='utf-8') as f:
        competition_desc = json.load(f)

    # Generate and return the naive solution
    naive_solution = naive_solution_gen(competition=competition_desc)
    
    # Define the output folder path
    script_dir = Path(__file__).parent  # Get the directory of this script
    output_folder = script_dir / competition_name

    # Create the folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)

    # Define the output file path
    output_file = output_folder / "naive_solution.txt"

    # Save the naive solution into the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(naive_solution))  # Convert to string in case it's not a string


# Example usage
competition_name = "optiver-trading-at-the-close"
competition_dir = Path("/data/userdata/share/kaggle")
naive_solution_generator(competition_name, competition_dir)
