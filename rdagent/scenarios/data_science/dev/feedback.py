import json
from pathlib import Path

from rdagent.components.knowledge_management.graph import UndirectedNode
from rdagent.core.experiment import Experiment
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T


class DSExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: DSExperiment, trace: DSTrace) -> HypothesisFeedback:
        hypothesis = exp.hypothesis
        current_results = exp.result
        if hypothesis.component == "DataLoadSpec":
            modified_file_name = "load_data.py"
        elif hypothesis.component == "FeatureEng":
            modified_file_name = "feature.py"
        elif hypothesis.component == "Model":
            modified_file_name = "model01.py"
        elif hypothesis.component == "Ensemble":
            modified_file_name = "ensemble.py"
        elif hypothesis.component == "Workflow":
            modified_file_name = "main.py"
        modified_code = exp.experiment_workspace.file_dict[modified_file_name]

        sota_hypothesis, sota_exp = trace.get_sota_hypothesis_and_experiment()

        if sota_exp:
            sota_codes = {
                "load_data.py": (sota_exp.experiment_workspace.workspace_path / "load_data.py").read_text(),
                "feature.py": (sota_exp.experiment_workspace.workspace_path / "feature.py").read_text(),
                "model.py": (sota_exp.experiment_workspace.workspace_path / "model.py").read_text(),
                "ensemble.py": (sota_exp.experiment_workspace.workspace_path / "ensemble.py").read_text(),
                "main.py": (sota_exp.experiment_workspace.workspace_path / "main.py").read_text(),
            }
            sota_results = sota_exp.result
        else:
            sota_codes = None
            sota_results = None

        last_hypothesis_and_feedback = None
        if trace.hist and len(trace.hist) > 0:
            last_hypothesis_and_feedback = (trace.hist[-1][0], trace.hist[-1][2])

        system_prompt = T(".prompts:exp_feedback.system").r(scenario=self.scen.get_scenario_all_desc())
        user_prompt = T(".prompts:exp_feedback.user").r(
            sota_codes=sota_codes,
            sota_results=sota_results,
            hypothesis=str(hypothesis),
            modified_code=modified_code,
            current_results=current_results,
            last_hypothesis_and_feedback=last_hypothesis_and_feedback,
        )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
        )

        return HypothesisFeedback(
            observations=resp_dict.get("Observations", "No observations provided"),
            hypothesis_evaluation=resp_dict.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=resp_dict.get("New Hypothesis", "No new hypothesis provided"),
            reason=resp_dict.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(resp_dict.get("Replace Best Result", "no")),
        )
