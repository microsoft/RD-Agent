import subprocess
from collections import defaultdict
from typing import Any

import pickle
import fire

from rdagent.oai.llm_utils import APIBackend

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.log.time import measure_time
from rdagent.scenarios.kaggle.experiment.utils import python_files_to_notebook
from rdagent.scenarios.kaggle.kaggle_crawler import download_data
from rdagent.scenarios.kaggle.proposal.proposal import (
    KG_ACTION_FEATURE_ENGINEERING,
    KG_ACTION_FEATURE_PROCESSING,
    KG_ACTION_MODEL_FEATURE_SELECTION,
    KGTrace,
)
from pathlib import Path
from jinja2 import Environment, StrictUndefined
from rdagent.utils.workflow import LoopBase, LoopMeta
from rdagent.core.prompts import Prompts

prompt_dict = Prompts(file_path=Path("./rdagent/app/kaggle/prompts.yaml"))

class KaggleBOLoop(LoopBase, metaclass=LoopMeta):
    @measure_time
    def __init__(self, PROP_SETTING: BasePropSetting):
        with logger.tag("init"):
            scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)
            logger.log_object(scen, tag="scenario")

            knowledge_base = (
                import_class(PROP_SETTING.knowledge_base)(PROP_SETTING.knowledge_base_path, scen)
                if PROP_SETTING.knowledge_base != ""
                else None
            )
            logger.log_object(knowledge_base, tag="knowledge_base")

            self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
            logger.log_object(self.hypothesis_gen, tag="hypothesis generator")

            self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
            logger.log_object(self.hypothesis2experiment, tag="hypothesis2experiment")

            self.feature_coder: Developer = import_class(PROP_SETTING.feature_coder)(scen)
            logger.log_object(self.feature_coder, tag="feature coder")
            self.model_feature_selection_coder: Developer = import_class(PROP_SETTING.model_feature_selection_coder)(
                scen
            )
            logger.log_object(self.model_feature_selection_coder, tag="model feature selection coder")
            self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
            logger.log_object(self.model_coder, tag="model coder")

            self.feature_runner: Developer = import_class(PROP_SETTING.feature_runner)(scen)
            logger.log_object(self.feature_runner, tag="feature runner")
            self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
            logger.log_object(self.model_runner, tag="model runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = KGTrace(scen=scen, knowledge_base=knowledge_base)
            super().__init__()

    @measure_time
    def propose(self, prev_out: dict[str, Any]):
        hypothesis_list = []
        for _ in range(2):
            hypothesis = self.hypothesis_gen.gen(self.trace)
            hypothesis_list.append(hypothesis)
        return hypothesis_list
    
    def _develop(self, hypothesis):
        with logger.tag("d"): # develop
            exp = self.hypothesis2experiment.convert(hypothesis, self.trace)
            if hypothesis.action in [KG_ACTION_FEATURE_ENGINEERING, KG_ACTION_FEATURE_PROCESSING]:
                code = self.feature_coder.develop(exp)
            elif hypothesis.action == KG_ACTION_MODEL_FEATURE_SELECTION:
                code = self.model_feature_selection_coder.develop(exp)
            else:
                code = self.model_coder.develop(exp)
        return code
    
    def _estimate(self, code):
        system_prompt = prompt_dict["System"] + "Here is the trace: " + Environment(undefined=StrictUndefined).from_string(prompt_dict["Trace Convert"]).render(trace=self.trace)
        user_prompt = "Here is the new implementation:" + str(code.sub_workspace_list[0].code) + \
        " Please evaluate its performance. Output a score between 0 and 1. Do not include anything else in your response."
        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt)
        return float(resp)
        
    @measure_time
    def sample(self, prev_out):
        codes = []
        for h in prev_out["propose"]:
            print(type(h))
            code = []
            for _ in range(2):
                c = self._develop(h)
                code.append(c)
            codes.append(code)
        return codes

    @measure_time
    def select(self, prev_out):
        results = []
        codes = []
        hs = []
        hypotheses = prev_out["propose"]
        exps = prev_out["sample"]
        for i in range(len(exps)):
            for j in range(len(exps[i])):
                codes.append(exps[i][j])
                r = self._estimate(exps[i][j])
                results.append(r)
                hs.append(hypotheses[i])
        m = max(results)
        index = results.index(m)
        logger.log_object(hs[index], tag="r.hypothesis generation")
        logger.log_object(codes[index].sub_workspace_list, tag="d.coder result")
        
        return codes[index], hs[index]

    # @measure_time
    # def exp_gen(self, prev_out: dict[str, Any]):
    #     with logger.tag("r"):  # research
    #         exp = self.hypothesis2experiment.convert(prev_out["propose"], self.trace)
    #         logger.log_object(exp.sub_tasks, tag="experiment generation")
    #     return exp

    # @measure_time
    # def coding(self, prev_out: dict[str, Any]):
    #     with logger.tag("d"):  # develop
    #         if prev_out["propose"].action in [KG_ACTION_FEATURE_ENGINEERING, KG_ACTION_FEATURE_PROCESSING]:
    #             exp = self.feature_coder.develop(prev_out["exp_gen"])
    #         elif prev_out["propose"].action == KG_ACTION_MODEL_FEATURE_SELECTION:
    #             exp = self.model_feature_selection_coder.develop(prev_out["exp_gen"])
    #         else:
    #             exp = self.model_coder.develop(prev_out["exp_gen"])
    #         logger.log_object(exp.sub_workspace_list, tag="coder result")
    #     return exp

    @measure_time
    def running(self, prev_out: dict[str, Any]):
        with logger.tag("ef"):  # evaluate and feedback
            if prev_out["propose"][0].action in [KG_ACTION_FEATURE_ENGINEERING, KG_ACTION_FEATURE_PROCESSING]:
                exp = self.feature_runner.develop(prev_out["select"][0])
            else:
                exp = self.model_runner.develop(prev_out["select"][0])
            logger.log_object(exp, tag="runner result")

            if KAGGLE_IMPLEMENT_SETTING.competition in ["optiver-realized-volatility-prediction"]:
                try:
                    python_files_to_notebook(
                        KAGGLE_IMPLEMENT_SETTING.competition, exp.experiment_workspace.workspace_path
                    )
                except Exception as e:
                    logger.error(f"Merge python files to one file failed: {e}")

            if KAGGLE_IMPLEMENT_SETTING.auto_submit:
                csv_path = exp.experiment_workspace.workspace_path / "submission.csv"
                try:
                    subprocess.run(
                        [
                            "kaggle",
                            "competitions",
                            "submit",
                            "-f",
                            str(csv_path.absolute()),
                            "-m",
                            str(csv_path.parent.absolute()),
                            KAGGLE_IMPLEMENT_SETTING.competition,
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Auto submission failed: \n{e}")
                except Exception as e:
                    logger.error(f"Other exception when use kaggle api:\n{e}")

        return exp

    @measure_time
    def feedback(self, prev_out: dict[str, Any]):
        feedback = self.summarizer.generate_feedback(prev_out["running"], prev_out["select"][1], self.trace)
        with logger.tag("ef"):  # evaluate and feedback
            logger.log_object(feedback, tag="feedback")
        self.trace.hist.append((prev_out["select"][1], prev_out["running"], feedback))
        # with open('trace.pkl', 'wb') as file:
        #     pickle.dump(self.trace, file)

    skip_loop_error = (ModelEmptyError, FactorEmptyError)


def main(path=None, step_n=None, competition=None):
    """
    Auto R&D Evolving loop for models in a kaggle{} scenario.

    You can continue running session by
    .. code-block:: bash

        dotenv run -- python rdagent/app/kaggle/bo_loop.py --competition playground-series-s4e8 [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional parameter
        rdagent kaggle --competition playground-series-s4e8  # You are encouraged to use this one.

    """
    if competition:
        KAGGLE_IMPLEMENT_SETTING.competition = competition
        download_data(competition=competition, local_path=KAGGLE_IMPLEMENT_SETTING.local_data_path)
    else:
        logger.error("Please specify competition name.")

    if path is None:
        kaggle_loop = KaggleBOLoop(KAGGLE_IMPLEMENT_SETTING)
    else:
        kaggle_loop = KaggleBOLoop.load(path)
    
    kaggle_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)