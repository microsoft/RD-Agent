import qlib
qlib.init()

from qlib.workflow import R
# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

print(123)

print(R.list_experiments())

# TODO: list all the recorder and metrics 

# Assuming you have already listed the experiments
experiments = R.list_experiments()

# Iterate through each experiment to list its recorders and metrics
for experiment in experiments:
    print(f"Experiment: {experiment}")
    recorders = R.list_recorders(experiment)
    for recorder_id in recorders:
        print(f"  Recorder ID: {recorder_id}")
        recorder = R.get_recorder(recorder_id, experiment)
        metrics = recorder.list_metrics()
        print(f"    Metrics: {metrics}")

# TODO: get the latest recorder
