# TODO: we should have more advanced mechanism to handle such requirements for saving sessions.
import json
import pickle
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, StrictUndefined

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
    load_and_process_pdfs_by_langchain,
)
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

assert load_dotenv()

scen: Scenario = import_class(FACTOR_PROP_SETTING.scen)()

hypothesis_gen: HypothesisGen = import_class(FACTOR_PROP_SETTING.hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(FACTOR_PROP_SETTING.hypothesis2experiment)()

qlib_factor_coder: Developer = import_class(FACTOR_PROP_SETTING.coder)(scen)

qlib_factor_runner: Developer = import_class(FACTOR_PROP_SETTING.runner)(scen)

qlib_factor_summarizer: HypothesisExperiment2Feedback = import_class(FACTOR_PROP_SETTING.summarizer)(scen)

with open(FACTOR_PROP_SETTING.report_result_json_file_path, "r") as f:
    judge_pdf_data = json.load(f)

prompts_path = Path(__file__).parent / "prompts.yaml"
prompts = Prompts(file_path=prompts_path)


def save_progress(trace, current_index):
    with open(FACTOR_PROP_SETTING.progress_file_path, "wb") as f:
        pickle.dump((trace, current_index), f)


def load_progress():
    if Path(FACTOR_PROP_SETTING.progress_file_path).exists():
        with open(FACTOR_PROP_SETTING.progress_file_path, "rb") as f:
            return pickle.load(f)
    return Trace(scen=scen), 0


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


trace, start_index = load_progress()

try:
    judge_pdf_data_items = list(judge_pdf_data.items())
    for index in range(start_index, len(judge_pdf_data_items)):
        if index > 1000:
            break
        file_path, attributes = judge_pdf_data_items[index]
        if attributes["class"] == 1:
            report_file_path = Path(
                file_path.replace(FACTOR_PROP_SETTING.origin_report_path, FACTOR_PROP_SETTING.local_report_path)
            )
            if report_file_path.exists():
                logger.info(f"Processing {report_file_path}")

                with logger.tag("r"):
                    exp, hypothesis = extract_factors_and_implement(str(report_file_path))
                    if exp is None:
                        continue
                    exp.based_experiments = [t[1] for t in trace.hist if t[2]]
                    if len(exp.based_experiments) == 0:
                        exp.based_experiments.append(QlibFactorExperiment(sub_tasks=[]))
                    logger.log_object(hypothesis, tag="hypothesis generation")
                    logger.log_object(exp.sub_tasks, tag="experiment generation")

                with logger.tag("d"):
                    exp = qlib_factor_coder.develop(exp)
                    logger.log_object(exp.sub_workspace_list)

                with logger.tag("ef"):
                    exp = qlib_factor_runner.develop(exp)
                    if exp is None:
                        logger.error(f"Factor extraction failed for {report_file_path}. Skipping to the next report.")
                        continue
                    logger.log_object(exp, tag="factor runner result")
                    feedback = qlib_factor_summarizer.generate_feedback(exp, hypothesis, trace)
                    logger.log_object(feedback, tag="feedback")

                trace.hist.append((hypothesis, exp, feedback))
                logger.info(f"Processed {report_file_path}: Result: {exp}")

                # Save progress after processing each report
                save_progress(trace, index + 1)
            else:
                logger.error(f"File not found: {report_file_path}")
except Exception as e:
    logger.error(f"An error occurred: {e}")
    save_progress(trace, index)
    raise
