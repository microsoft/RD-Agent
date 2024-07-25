import json
from pathlib import Path
import pickle
from dotenv import load_dotenv
from jinja2 import Environment, StrictUndefined
import pandas as pd

from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.components.document_reader.document_reader import load_and_process_pdfs_by_langchain
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.developer.factor_coder import QlibFactorCoSTEER
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario, QlibFactorExperiment
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
    classify_report_from_dict,
)

from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Hypothesis,
    Trace,
)

from rdagent.core.developer import Developer

assert load_dotenv()

scen: Scenario = import_class(PROP_SETTING.factor_scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.factor_hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.factor_hypothesis2experiment)()

qlib_factor_coder: Developer = import_class(PROP_SETTING.factor_coder)(scen)

qlib_factor_runner: Developer = import_class(PROP_SETTING.factor_runner)(scen)

qlib_factor_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.factor_summarizer)(scen)

with open(PROP_SETTING.report_result_json_file_path, 'r') as f:
    judge_pdf_data = json.load(f)

prompts_path = Path(__file__).parent / "prompts.yaml"
prompts = Prompts(file_path=prompts_path)

def generate_hypothesis(factor_result: dict, report_content: str) -> str:
    system_prompt = Environment(undefined=StrictUndefined).from_string(prompts["hypothesis_generation"]["system"]).render()
    user_prompt = Environment(undefined=StrictUndefined).from_string(prompts["hypothesis_generation"]["user"]).render(
        factor_descriptions=json.dumps(factor_result),
        report_content=report_content
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
    )

    response_json = json.loads(response)
    hypothesis_text = response_json.get("hypothesis", "No hypothesis generated.")
    reason_text = response_json.get("reason", "No reason provided.")

    return Hypothesis(hypothesis=hypothesis_text, reason=reason_text)

def extract_factors_and_implement(report_file_path: str) -> tuple:
    scenario = QlibFactorScenario()

    with logger.tag("extract_factors_and_implement"):
        with logger.tag("load_factor_tasks"):

            exp = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
            if exp is None or exp.sub_tasks == []:
                return None, None
            
    docs_dict = load_and_process_pdfs_by_langchain(Path(report_file_path))

    factor_result = {
        task.factor_name: {
            "description": task.factor_description,
            "formulation": task.factor_formulation,
            "variables": task.variables,
            "resources": task.factor_resources
        }
        for task in exp.sub_tasks
    }

    report_content = "\n".join(docs_dict.values())
    hypothesis = generate_hypothesis(factor_result, report_content)

    return exp, hypothesis

trace = Trace(scen=scen)

for file_path, attributes in judge_pdf_data.items():
    if attributes["class"] == 1:
        report_file_path = Path(file_path.replace(PROP_SETTING.origin_report_path, PROP_SETTING.local_report_path))
        if report_file_path.exists():
            logger.info(f"Processing {report_file_path}")
            exp, hypothesis = extract_factors_and_implement(str(report_file_path))
            if exp is None:
                continue
            exp.based_experiments = [t[1] for t in trace.hist if t[2]]
            if len(exp.based_experiments) == 0:
                exp.based_experiments.append(QlibFactorExperiment(sub_tasks=[]))
            exp = qlib_factor_coder.develop(exp)
            exp = qlib_factor_runner.develop(exp)
            if exp is None:
                logger.error(f"Factor extraction failed for {report_file_path}. Skipping to the next report.")
                continue
            feedback = qlib_factor_summarizer.generate_feedback(exp, hypothesis, trace)

            trace.hist.append((hypothesis, exp, feedback))
            logger.info(f"Processed {report_file_path}: Result: {exp}")
        else:
            logger.error(f"File not found: {report_file_path}")
