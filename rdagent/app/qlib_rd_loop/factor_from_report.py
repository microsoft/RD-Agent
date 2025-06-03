import json
from pathlib import Path
from typing import Any, Dict, Tuple

import fire

from rdagent.app.qlib_rd_loop.conf import FACTOR_FROM_REPORT_PROP_SETTING
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
    load_and_process_pdfs_by_langchain,
)
from rdagent.core.proposal import Hypothesis
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import LoopMeta


def generate_hypothesis(factor_result: dict, report_content: str) -> str:
    """
    Generate a hypothesis based on factor results and report content.

    Args:
        factor_result (dict): The results of the factor analysis.
        report_content (str): The content of the report.

    Returns:
        str: The generated hypothesis.
    """
    system_prompt = T(".prompts:hypothesis_generation.system").r()
    user_prompt = T(".prompts:hypothesis_generation.user").r(
        factor_descriptions=json.dumps(factor_result), report_content=report_content
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        json_target_type=Dict[str, str],
    )

    response_json = json.loads(response)

    return Hypothesis(
        hypothesis=response_json.get("hypothesis", "No hypothesis provided"),
        reason=response_json.get("reason", "No reason provided"),
        concise_reason=response_json.get("concise_reason", "No concise reason provided"),
        concise_observation=response_json.get("concise_observation", "No concise observation provided"),
        concise_justification=response_json.get("concise_justification", "No concise justification provided"),
        concise_knowledge=response_json.get("concise_knowledge", "No concise knowledge provided"),
    )


def extract_hypothesis_and_exp_from_reports(report_file_path: str) -> Tuple[QlibFactorExperiment, Hypothesis]:
    """
    Extract hypothesis and experiment details from report files.

    Args:
        report_file_path (str): Path to the report file.

    Returns:
        Tuple[QlibFactorExperiment, Hypothesis]: The extracted experiment and generated hypothesis.
    """
    with logger.tag("extract_factors_and_implement"):
        with logger.tag("load_factor_tasks"):
            exp = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
            if exp is None or exp.sub_tasks == []:
                return None, None

        with logger.tag("load_pdf_screenshot"):
            pdf_screenshot = extract_first_page_screenshot_from_pdf(report_file_path)
            logger.log_object(pdf_screenshot)

    docs_dict = load_and_process_pdfs_by_langchain(report_file_path)

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
    exp.hypothesis = hypothesis
    return exp, hypothesis


class FactorReportLoop(FactorRDLoop, metaclass=LoopMeta):
    def __init__(self, report_folder: str = None):
        super().__init__(PROP_SETTING=FACTOR_FROM_REPORT_PROP_SETTING)
        if report_folder is None:
            self.judge_pdf_data_items = json.load(
                open(FACTOR_FROM_REPORT_PROP_SETTING.report_result_json_file_path, "r")
            )
        else:
            self.judge_pdf_data_items = [i for i in Path(report_folder).rglob("*.pdf")]

        self.pdf_file_index = 0
        self.valid_pdf_file_count = 0
        self.current_loop_hypothesis = None
        self.current_loop_exp = None
        self.steps = ["propose_hypo_exp", "propose", "direct_exp_gen", "coding", "running", "feedback"]

    def propose_hypo_exp(self, prev_out: dict[str, Any]):
        with logger.tag("r"):
            while True:
                if FACTOR_FROM_REPORT_PROP_SETTING.is_report_limit_enabled and self.valid_pdf_file_count > 15:
                    break
                report_file_path = self.judge_pdf_data_items[self.pdf_file_index]
                logger.info(f"Processing number {self.pdf_file_index} report: {report_file_path}")
                self.pdf_file_index += 1
                exp, hypothesis = extract_hypothesis_and_exp_from_reports(str(report_file_path))
                if exp is None:
                    continue
                self.valid_pdf_file_count += 1
                exp.based_experiments = [QlibFactorExperiment(sub_tasks=[], hypothesis=hypothesis)] + [
                    t[0] for t in self.trace.hist if t[1]
                ]
                exp.sub_workspace_list = exp.sub_workspace_list[: FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp]
                exp.sub_tasks = exp.sub_tasks[: FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp]
                logger.log_object(hypothesis, tag="hypothesis generation")
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                self.current_loop_hypothesis = hypothesis
                self.current_loop_exp = exp
                return None

    def propose(self, prev_out: dict[str, Any]):
        return self.current_loop_hypothesis

    def direct_exp_gen(self, prev_out: dict[str, Any]):
        return {"propose": self.current_loop_hypothesis, "exp_gen": self.current_loop_exp}

    def coding(self, prev_out: dict[str, Any]):
        with logger.tag("d"):  # develop
            exp = self.coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
            logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp


def main(report_folder=None, path=None, step_n=None):
    """
    Auto R&D Evolving loop for fintech factors (the factors are extracted from finance reports).

    Args:
        report_folder (str, optional): The folder contains the report PDF files. Reports will be loaded from this folder.
        path (str, optional): The path for loading a session. If provided, the session will be loaded.
        step_n (int, optional): Step number to continue running a session.
    """
    if path is None and report_folder is None:
        model_loop = FactorReportLoop()
    elif path is not None:
        model_loop = FactorReportLoop.load(path)
    else:
        model_loop = FactorReportLoop(report_folder=report_folder)

    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
