from pathlib import Path
import qlib
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import pickle
import os

qlib.init()

from qlib.workflow import R
# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

# TODO: list all the recorder and metrics 

# Assuming you have already listed the experiments
experiments = R.list_experiments()

# Iterate through each experiment to find the latest recorder
experiment_name = None
latest_recorder = None
for experiment in experiments:
    # print(f"Experiment: {experiment}")
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            end_time = recorder.info['end_time']
            if latest_recorder is None or end_time > latest_recorder.info['end_time']:
                latest_recorder = recorder

# Check if the latest recorder is found
if latest_recorder is None:
    print("No recorders found")
else:
    print(f"Latest recorder: {latest_recorder}")

    # Load the specified file from the latest recorder
    file_path = "portfolio_analysis/port_analysis_1day.pkl"
    indicator_analysis_df = latest_recorder.load_object(file_path)

    # Optionally convert to DataFrame if not already in DataFrame format
    if not isinstance(indicator_analysis_df, pd.DataFrame):
        indicator_analysis_df = pd.DataFrame(indicator_analysis_df)
    
    output_path = os.path.join(str(Path(__file__).resolve().parent), "qlib_res.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(indicator_analysis_df, f)

    print("here2")
    print(output_path)