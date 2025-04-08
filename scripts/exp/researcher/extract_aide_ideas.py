import json
import os
from pathlib import Path
from rdagent.utils.agent.tpl import T
from scripts.exp.researcher.utils import solution_to_idea, extract_JSON

# Path to the aide solution
SOLUTION_PATH = "/data/userdata/v-yuanteli/aide_gpt_o1_our_results_best_solution/nomad2018-predict-transparent-conductors_0.06262_6.0295790671217295/solution.py"  
OUTPUT_DIR = "/data/userdata/v-xhong/researcher_u/RD-Agent/scripts/exp/researcher/output_dir/idea_pool"
OUTPUT_FILE = "aide_ideas.json"

def main():
    # Read the solution file
    with open(SOLUTION_PATH, "r") as f:
        solution = f.read()
    
    # Get component description
    component_desc = "\\n".join(
        [
            f"[{key}] {value}"
            for key, value in T("scenarios.data_science.share:component_description").template.items()
        ]
    )
    
    # Get competition description 
    competition_desc = """
    # Competition: nomad2018-predict-transparent-conductors
    
    The goal of this competition is to identify the material compositions most likely to lead to high-performance transparent conductors. The dataset contains information about various material compositions and their properties, including formation energy and bandgap energy.
    
    # Data Description
    
    The dataset includes features such as:
    - Lattice vectors and angles
    - Atomic compositions (percentages of Al, Ga, In, and O)
    - Spacegroup
    - Number of atoms
    
    The target variables are:
    - formation_energy_ev_natom: Formation energy per atom (eV)
    - bandgap_energy_ev: Bandgap energy (eV)
    
    # Evaluation
    
    The evaluation metric is the mean of the root mean squared logarithmic error (RMSLE) for both target variables.
    """
    
    # Extract ideas from the solution
    raw_ideas = solution_to_idea(component_desc, competition_desc, solution)
    ideas = extract_JSON(raw_ideas)
    
    # Save the ideas to a JSON file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(ideas, f, indent=2)
    
    print(f"Extracted {len(ideas)} ideas from the aide solution and saved to {output_path}")

if __name__ == "__main__":
    main()