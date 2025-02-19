import qlib
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

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
end_times = {key: value.info["end_time"] for key, value in recorder_list.items()}
sorted_end_times = dict(sorted(end_times.items(), key=lambda item: item[1], reverse=True))

latest_recorder_id = next(iter(sorted_end_times))
print(f"Latest recorder ID: {latest_recorder_id}")

latest_recorder = R.get_recorder(experiment_name=experiment_name, recorder_id=latest_recorder_id)
print(f"Latest recorder: {latest_recorder}")

pred_df = latest_recorder.load_object("pred.pkl")
print("pred_df", pred_df)

ic_df = latest_recorder.load_object("sig_analysis/ic.pkl")
print("ic_df: ", ic_df)

ric_df = latest_recorder.load_object("sig_analysis/ric.pkl")
print("ric_df: ", ric_df)

print("list_metrics: ", latest_recorder.list_metrics())
print("IC: ", latest_recorder.list_metrics()["IC"])
print("ICIR: ", latest_recorder.list_metrics()["ICIR"])
print("Rank IC: ", latest_recorder.list_metrics()["Rank IC"])
print("Rank ICIR: ", latest_recorder.list_metrics()["Rank ICIR"])
