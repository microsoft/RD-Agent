import os
from pathlib import Path
from typing import List, Tuple, Any
import pandas as pd

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.core.exception import FactorEmptyError

def run_factor_file(factor_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Run a single factor file and return its results
    """
    try:

        import pickle

        # Define the file path
        file_path = "/data/userdata/v-yuanteli/RD-Agent/git_ignore_folder/RD-Agent_workspace/90454c509f6547a7a6000f69e94ff63c/fac4o/1.pkl"

        # Load the pickle file
        with open(file_path, "rb") as file:
            fac4o_data = pickle.load(file)
        with open(factor_path, "r") as factor_file:
            factor_content = factor_file.read()
        fac4o_data.file_dict['factor.py'] = factor_content
        # Execute the factor
        if hasattr(fac4o_data, 'execute'):
            print(f"Executing {factor_path}")
            result, df = fac4o_data.execute("All")
            if isinstance(df, pd.DataFrame):
                return f"Successfully executed {factor_path}", df
            else:
                return f"Result from {factor_path} is not a DataFrame", None
        else:
            return f"No execute method found in {factor_path}", None
    except Exception as e:
        return f"Error executing {factor_path}: {str(e)}", None

def run_all_factors(factors_dir: str) -> pd.DataFrame:
    """
    Run all factor*.py files in the specified directory and combine their results
    """
    # Get all factor*.py files
    factor_files = []
    for root, _, files in os.walk(factors_dir):
        for file in files:
            if file.startswith("factor") and file.endswith(".py"):
                factor_files.append(os.path.join(root, file))
    
    if not factor_files:
        logger.warning(f"No factor*.py files found in {factors_dir}")
        raise FactorEmptyError("No factor files found to execute")
    
    logger.info(f"Found {len(factor_files)} factor files to execute")
    
    # Run all factors in parallel
    results = [run_factor_file(factor_path) for factor_path in factor_files]
    
    # Collect all successful factor DataFrames
    factor_dfs = []
    for message, result in results:
        logger.info(message)
        if result is not None and isinstance(result, pd.DataFrame):
            factor_dfs.append(result)
    
    # Combine all successful factor data
    if factor_dfs:
        combined_df = pd.concat(factor_dfs, axis=1)
        # Sort and remove duplicate columns
        combined_df = combined_df.sort_index()
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep="last")]
        return combined_df
    else:
        raise FactorEmptyError("No valid factor data found to merge")

if __name__ == "__main__":
    # Specify the directory containing factor files
    factors_directory = "/data/userdata/v-yuanteli/RD-Agent/git_ignore_folder/RD-Agent_workspace/90454c509f6547a7a6000f69e94ff63c/fac4o"  # Replace with your actual directory path
    
    try:
        # Run all factors and get combined DataFrame
        combined_df = run_all_factors(factors_directory)
        
        # Save the combined DataFrame
        output_path = os.path.join(factors_directory, "combined_factors.parquet")
        combined_df.to_parquet(output_path, engine="pyarrow")
        logger.info(f"Combined factors saved to {output_path}")
        
        # Print some basic information about the combined DataFrame
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        logger.info(f"Columns: {combined_df.columns.tolist()}")
        
    except FactorEmptyError as e:
        logger.error(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}") 