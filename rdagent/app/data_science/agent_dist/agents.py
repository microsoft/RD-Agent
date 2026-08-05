
import jsonpickle
from conf import DIST_SETTING
from flask import Flask, jsonify, request
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.proposal.exp_gen import DSExpGen

from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.feature import FeatureCoSTEER
from rdagent.components.coder.data_science.model import ModelCoSTEER
from rdagent.components.coder.data_science.pipeline import PipelineCoSTEER
from rdagent.components.coder.data_science.raw_data_loader import DataLoaderCoSTEER
from rdagent.components.coder.data_science.workflow import WorkflowCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.proposal import ExperimentFeedback

app = Flask(__name__)

@app.route("/exp-gen", methods=["POST"])
def exp_gen():
    """Research"""
    data = request.get_json()
    try:
        # Decode the provided jsonpickled objects.
        scen = jsonpickle.decode(data["scen"])
        trace = jsonpickle.decode(data["trace"])
        exp = DSExpGen(scen).gen(trace)
        # Serialize the experiment object using jsonpickle.
        exp_pickle = jsonpickle.encode(exp, unpicklable=True)
        return jsonify({"experiment": exp_pickle}), 200
    except Exception as e:
        return jsonify({"error": jsonpickle.encode(e)}), 500



@app.route("/coding", methods=["POST"])
def coding():
    data = request.get_json()
    try:
        # Decode the provided jsonpickled objects.
        scen = jsonpickle.decode(data["scen"])
        exp = jsonpickle.decode(data["exp"])
        # Initialize coders
        data_loader_coder = DataLoaderCoSTEER(scen)
        feature_coder = FeatureCoSTEER(scen)
        model_coder = ModelCoSTEER(scen)
        ensemble_coder = EnsembleCoSTEER(scen)
        workflow_coder = WorkflowCoSTEER(scen)
        pipeline_coder = PipelineCoSTEER(scen)

        # Process tasks
        for tasks in exp.pending_tasks_list:
            exp.sub_tasks = tasks
            with logger.tag(f"{exp.sub_tasks[0].__class__.__name__}"):
                if isinstance(exp.sub_tasks[0], DataLoaderTask):
                    exp = data_loader_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], FeatureTask):
                    exp = feature_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], ModelTask):
                    exp = model_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], EnsembleTask):
                    exp = ensemble_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], WorkflowTask):
                    exp = workflow_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], PipelineTask):
                    exp = pipeline_coder.develop(exp)
                else:
                    raise NotImplementedError(f"Unsupported component in DataScienceRDLoop: {exp.hypothesis.component}")
            exp.sub_tasks = []

        # Serialize the updated experiment object using jsonpickle.
        exp_pickle = jsonpickle.encode(exp, unpicklable=True)
        return jsonify({"experiment": exp_pickle}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": jsonpickle.encode(e)}), 500


@app.route("/run", methods=["POST"])
def run():
    """Run the experiment"""
    data = request.get_json()
    try:
        # Decode the provided jsonpickled objects.
        scen = jsonpickle.decode(data["scen"])
        exp = jsonpickle.decode(data["exp"])
        
        # Initialize the runner
        runner = DSCoSTEERRunner(scen)
        
        # Develop the experiment using the runner
        new_exp = runner.develop(exp)
        
        # Serialize the updated experiment object using jsonpickle.
        exp_pickle = jsonpickle.encode(new_exp, unpicklable=True)
        return jsonify({"experiment": exp_pickle}), 200
    except Exception as e:
        return jsonify({"error": jsonpickle.encode(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """Generate feedback for the experiment"""
    data = request.get_json()
    try:
        # Decode the provided jsonpickled objects.
        scen = jsonpickle.decode(data["scen"])
        exp = jsonpickle.decode(data["exp"])
        trace = jsonpickle.decode(data["trace"])
        
        # Initialize the summarizer
        summarizer = DSExperiment2Feedback(scen)
        
        # Generate feedback using the summarizer

        if trace.next_incomplete_component() is None or DS_RD_SETTING.coder_on_whole_pipeline:
            # we have alreadly completed components in previous trace. So current loop is focusing on a new proposed idea.
            # So we need feedback for the proposal.
            feedback = summarizer.generate_feedback(exp, trace)
        else:
            # Otherwise, it is on drafting stage, don't need complicated feedbacks.
            feedback = ExperimentFeedback(
                reason=f"{exp.hypothesis.component} is completed.",
                decision=True,
            )
        
        # Serialize the feedback object using jsonpickle.
        feedback_pickle = jsonpickle.encode(feedback, unpicklable=True)
        return jsonify({"feedback": feedback_pickle}), 200
    except Exception as e:
        return jsonify({"error": jsonpickle.encode(e)}), 500


if __name__ == "__main__":
    app.run(host=DIST_SETTING.host, port=DIST_SETTING.port, debug=False)
