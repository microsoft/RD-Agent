import qlib
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
qlib.init()

from qlib.workflow import R
# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

# TODO: list all the recorder and metrics 

# Assuming you have already listed the experiments
experiments = R.list_experiments()

# Iterate through each experiment to list its recorders and metrics
experiment_name = None
for experiment in experiments:
    print(f"Experiment: {experiment}")
    recorders = R.list_recorders(experiment_name=experiment)
    # print(recorders)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
        print(f"Recorder ID: {recorder_id}")
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
        metrics = recorder.list_metrics()
        print(f"Metrics: {metrics}")

# TODO: get the latest recorder

recorder_list = R.list_recorders(experiment_name="workflow")
end_times = {key: value.info['end_time'] for key, value in recorder_list.items()}
sorted_end_times = dict(sorted(end_times.items(), key=lambda item: item[1], reverse=True))
latest_recorder_id = next(iter(sorted_end_times))
print(f"Latest recorder ID: {latest_recorder_id}")
latest_recorder = R.get_recorder(experiment_name=experiment_name, recorder_id=latest_recorder_id)
print(f"Latest recorder: {latest_recorder}")
metrics = latest_recorder.list_metrics()
print(f"Latest Metrics: {metrics}")
