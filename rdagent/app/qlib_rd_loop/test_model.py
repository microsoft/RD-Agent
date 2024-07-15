import pickle
from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.task_generator import TaskGenerator
from rdagent.core.utils import import_class

# Initialize the necessary components
scen: Scenario = import_class(PROP_SETTING.qlib_model_scen)()
hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.qlib_model_hypothesis_gen)(scen)
hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.qlib_model_hypothesis2experiment)()
qlib_model_coder: TaskGenerator = import_class(PROP_SETTING.qlib_model_coder)(scen)
qlib_model_runner: TaskGenerator = import_class(PROP_SETTING.qlib_model_runner)(scen)
qlib_model_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.qlib_model_summarizer)()

# Function to dump exp, hypothesis, and trace objects
def dump_objects(iteration: int):
    # Create a trace object
    trace = Trace(scen=scen)

    for _ in range(iteration):
        # Generate hypothesis and experiment
        hypothesis = hypothesis_gen.gen(trace)
        exp = hypothesis2experiment.convert(hypothesis, trace)
        exp = qlib_model_coder.generate(exp)

        # Dump the exp, hypothesis, and trace objects to a pickle file
        with open(f'dumped_objects_{_}.pkl', 'wb') as f:
            pickle.dump((exp, hypothesis, trace), f)


# Function to test with dumped objects
def test_with_dumped_objects(iteration: int):
    # Load the pickle file
    with open(f'dumped_objects_{iteration}.pkl', 'rb') as f:
        exp, hypothesis, trace = pickle.load(f)

    # Run the required lines
    exp = qlib_model_runner.generate(exp)
    feedback = qlib_model_summarizer.generateFeedback(exp, hypothesis, trace)

    # Print the feedback to verify
    print("Generated Decision:", feedback.decision)
    print("Observations:", feedback.observations)
    print("Hypothesis Evaluation:", feedback.hypothesis_evaluation)
    print("New Hypothesis:", feedback.new_hypothesis)
    print("Reason:", feedback.reason)

    # Append the results to the trace history
    trace.hist.append((hypothesis, exp, feedback))

# Example usage
if __name__ == "__main__":
    # Dump objects for a specific iteration
    dump_objects(1)  # Adjust the iteration count as needed

    # Test with the dumped objects
    test_with_dumped_objects(0)  

    # with open(f'dumped_objects_{0}.pkl', 'rb') as f:
    #     exp, hypothesis, trace = pickle.load(f)

    # print(exp.sub_implementations[0].code_dict['model.py'])
