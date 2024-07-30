# TODO: we should have more advanced mechanism to handle such requirements for saving sessions.
import json
import pickle
from pathlib import Path
from typing import Any

import fire
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, StrictUndefined

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
    load_and_process_pdfs_by_langchain,
)
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.developer import Developer
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import (
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.developer.factor_coder import QlibFactorCoSTEER
from rdagent.scenarios.qlib.experiment.factor_experiment import (
    QlibFactorExperiment,
    QlibFactorScenario,
)
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
    classify_report_from_dict,
)
from rdagent.utils.workflow import LoopBase, LoopMeta

with open(FACTOR_PROP_SETTING.report_result_json_file_path, "r") as f:
    judge_pdf_data = json.load(f)

prompts_path = Path(__file__).parent / "prompts.yaml"
prompts = Prompts(file_path=prompts_path)



def generate_hypothesis(factor_result: dict, report_content: str) -> str:
    system_prompt = (
        Environment(undefined=StrictUndefined).from_string(prompts["hypothesis_generation"]["system"]).render()
    )
    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompts["hypothesis_generation"]["user"])
        .render(factor_descriptions=json.dumps(factor_result), report_content=report_content)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
    )

    response_json = json.loads(response)
    hypothesis_text = response_json.get("hypothesis", "No hypothesis generated.")
    reason_text = response_json.get("reason", "No reason provided.")
    concise_reason_text = response_json.get("concise_reason", "No concise reason provided.")
    concise_reason_text = response_json.get("concise_reason", "No concise reason provided.")

    return Hypothesis(hypothesis=hypothesis_text, reason=reason_text, concise_reason=concise_reason_text)
    return Hypothesis(hypothesis=hypothesis_text, reason=reason_text, concise_reason=concise_reason_text)


def extract_factors_and_implement(report_file_path: str) -> tuple:
    scenario = QlibFactorScenario()

    with logger.tag("extract_factors_and_implement"):
        with logger.tag("load_factor_tasks"):
            exp = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
            if exp is None or exp.sub_tasks == []:
                return None, None

        with logger.tag("load_pdf_screenshot"):
            pdf_screenshot = extract_first_page_screenshot_from_pdf(report_file_path)
            logger.log_object(pdf_screenshot)

    docs_dict = load_and_process_pdfs_by_langchain(Path(report_file_path))

    factor_result = {
        task.factor_name: {
            "description": task.factor_description,
            "formulation": task.factor_formulation,
            "variables": task.variables,
            "resources": task.factor_resources,
        }
        for task in exp.sub_tasks
    }

    report_content = "\n".join(docs_dict.values())
    hypothesis = generate_hypothesis(factor_result, report_content)

    return exp, hypothesis


class FactorReportLoop(LoopBase, metaclass=LoopMeta):
    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()

        self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
        self.runner: Developer = import_class(PROP_SETTING.runner)(scen)

        self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
        self.trace = Trace(scen=scen)

        self.judge_pdf_data_items = list(judge_pdf_data.items())
        self.index = 0
        super().__init__()

    def propose_hypo_exp(self, prev_out: dict[str, Any]):
        with logger.tag("r"):
            while True:
                file_path, attributes = self.judge_pdf_data_items[self.index]
                self.index += 1
                if attributes["class"] == 1:
                    report_file_path = Path(
                        file_path.replace(FACTOR_PROP_SETTING.origin_report_path, FACTOR_PROP_SETTING.local_report_path)
                    )
                    if report_file_path.exists():
                        logger.info(f"Processing {report_file_path}")
                    exp, hypothesis = extract_factors_and_implement(str(report_file_path))
                    if exp is None:
                        continue
                    exp.based_experiments = [t[1] for t in self.trace.hist if t[2]]
                    if len(exp.based_experiments) == 0:
                        exp.based_experiments.append(QlibFactorExperiment(sub_tasks=[]))
                    logger.log_object(hypothesis, tag="hypothesis generation")
                    logger.log_object(exp.sub_tasks, tag="experiment generation")
                    return hypothesis,  exp

    def coding(self, prev_out: dict[str, Any]):
        with logger.tag("d"):  # develop
            exp = self.coder.develop(prev_out["propose_hypo_exp"][1])
            logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        with logger.tag("ef"):  # evaluate and feedback
            exp = self.runner.develop(prev_out["coding"])
            logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        feedback = self.summarizer.generate_feedback(prev_out["running"], prev_out["propose_hypo_exp"][0], self.trace)
        with logger.tag("ef"):  # evaluate and feedback
            logger.log_object(feedback, tag="feedback")
        self.trace.hist.append((prev_out["propose_hypo_exp"][0], prev_out["running"], feedback))

def main(path=None, step_n=None):
    """
    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor_w_sc.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if path is None:
        model_loop = FactorReportLoop(FACTOR_PROP_SETTING)
    else:
        model_loop = FactorReportLoop.load(path)
    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
