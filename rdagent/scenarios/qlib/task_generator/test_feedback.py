import json
from rdagent.oai.llm_utils import APIBackend
from rdagent.core.proposal import HypothesisExperiment2Feedback, Trace, Hypothesis, HypothesisFeedback, Scenario
from rdagent.core.experiment import Experiment
from typing import Dict, List, Tuple, Sequence
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment, QlibModelScenario
from rdagent.scenarios.qlib.task_generator.feedback import QlibFactorExperiment2Feedback

# Test the implementation

# Create a mock scenario
scenario = QlibModelScenario()

# Create a mock hypothesis
hypothesis = Hypothesis(
    hypothesis="The data shows time-series quality.",
    reason="This experiment is aimed at improving time-series prediction accuracy."
)

# Example results for each task
result_task1 = {
    "mean": 0.0161,
    "std": 1.4384,
    "annualized_return": 3.8237,
    "information_ratio": 0.1723,
    "max_drawdown": -48.7617,
    "observations": "The current benchmark return has lower mean and annualized returns. Volatility (std) and max drawdown have increased, indicating higher risk. Information ratio has significantly decreased, suggesting reduced efficiency."
}

result_task2 = {
    "mean": 0.0265,
    "std": 0.4131,
    "annualized_return": 6.3043,
    "information_ratio": 0.9892,
    "max_drawdown": -8.4109,
    "observations": "Decrease in mean and annualized return. Lower standard deviation, indicating reduced volatility. Decreased information ratio, although still relatively high. Slight increase in max drawdown."
}

# Create mock tasks for the experiment
sub_tasks = ["task1", "task2"]

# Initialize the experiment correctly
experiment = QlibModelExperiment(sub_tasks=sub_tasks)
experiment.result = [result_task1, result_task2]  # Store the results

# Create a mock trace and add a history entry
trace = Trace(scen=scenario)
trace.hist.append((hypothesis, experiment, HypothesisFeedback(
    observations="Initial observation based on the new model's performance metrics.",
    hypothesis_evaluation="The hypothesis shows mixed results under current conditions.",
    new_hypothesis="Investigate the factors leading to performance variations and refine the approach.",
    reason="The current results exhibit decreased performance compared to previous benchmarks.",
    decision=False
)))

# Example usage
if __name__ == "__main__":
    # Create the QlibFactorExperiment2Feedback object
    feedback_generator = QlibFactorExperiment2Feedback()
    feedback = feedback_generator.generateFeedback(experiment, hypothesis, trace)
    # Print the generated feedback
    print("Generated Decision:", feedback.decision)
    print("Observations:", feedback.observations)
    print("Hypothesis Evaluation:", feedback.hypothesis_evaluation)
    print("New Hypothesis:", feedback.new_hypothesis)
    print("Reason:", feedback.reason)

