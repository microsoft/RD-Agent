from pathlib import Path
from typing import Any

import fire
import jsonpickle
import requests
from rdagent.app.data_science.agent_dist.conf import DIST_SETTING
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import CoderError, RunnerError
from rdagent.core.proposal import ExperimentFeedback
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.scenarios.kaggle.kaggle_crawler import download_data


class DataScienceRDLoop(RDLoop):
    skip_loop_error = (CoderError, RunnerError)

    def __init__(self, PROP_SETTING: BasePropSetting):
        logger.log_object(PROP_SETTING.competition, tag="competition")
        self.scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)
        self.trace = DSTrace(scen=self.scen)

        super(RDLoop, self).__init__()

    def _call_api(self, get_key, uri, **kwargs):
        # Make a POST request to the exp-gen endpoint
        # print({k: jsonpickle.encode(v, unpicklable=True) for k, v in kwargs.items()})
        response = requests.post(
            f"http://{DIST_SETTING.host}:{DIST_SETTING.port}/{uri}",
            json={k: jsonpickle.encode(v, unpicklable=True) for k, v in kwargs.items()},
        )

        # Check if the request was successful
        exp_data = response.json()
        if response.status_code == 200:
            return jsonpickle.decode(exp_data[get_key])
        else:
            print(f"Failed to generate experiment: {response.json()['error']}")
            raise jsonpickle.decode(exp_data["error"])

    def direct_exp_gen(self, prev_out: dict[str, Any]):
        # Call exp_gen to generate a new experiment
        # Serialize the scenario and trace using jsonpickle
        return self._call_api("experiment", "exp-gen", **{"scen": self.scen, "trace": self.trace})

    def coding(self, prev_out: dict[str, Any]):
        exp = prev_out["direct_exp_gen"]
        exp = self._call_api("experiment", "coding", **{"exp": exp, "scen": self.scen})
        logger.log_object(exp)
        return exp

    def running(self, prev_out: dict[str, Any]):
        exp: DSExperiment = prev_out["coding"]
        if exp.is_ready_to_run():
            new_exp = self._call_api("experiment", "run", **{"exp": exp, "scen": self.scen})
            logger.log_object(new_exp)
            return new_exp
        return exp

    def feedback(self, prev_out: dict[str, Any]) -> ExperimentFeedback:
        exp: DSExperiment = prev_out["running"]
        feedback = self._call_api("feedback", "feedback", **{"exp": exp, "scen": self.scen, "trace": self.trace})
        logger.log_object(feedback)
        return feedback

    def record(self, prev_out: dict[str, Any]):
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is None:
            self.trace.hist.append((prev_out["running"], prev_out["feedback"]))
        else:
            self.trace.hist.append(
                (
                    prev_out["direct_exp_gen"] if isinstance(e, CoderError) else prev_out["coding"],
                    ExperimentFeedback.from_exception(e),
                )
            )
            if (
                self.trace.sota_experiment() is None
                and len(self.trace.hist) >= DS_RD_SETTING.consecutive_errors
                and not DS_RD_SETTING.coder_on_whole_pipeline
            ):
                # if {in inital/drafting stage} and {tried enough times}
                for _, fb in self.trace.hist[-DS_RD_SETTING.consecutive_errors :]:
                    if fb:
                        break  # any success will stop restarting.
                else:  # otherwise restart it
                    logger.error("Consecutive errors reached the limit. Dumping trace.")
                    logger.log_object(self.trace, tag="trace before restart")
                    self.trace = DSTrace(scen=self.trace.scen, knowledge_base=self.trace.knowledge_base)
        logger.log_object(self.trace, tag="trace")
        logger.log_object(self.trace.sota_experiment(), tag="SOTA experiment")


def main(
    path=None, output_path=None, step_n=None, loop_n=None, competition="bms-molecular-translation", do_truncate=True
):
    """

    Parameters
    ----------
    path :
        path like `$LOG_PATH/__session__/1/0_propose`. It indicates that we restore the state that after finish the step 0 in loop 1
    output_path :
        path like `$LOG_PATH`. It indicates that where we want to save our session and log information.
    step_n :
        How many steps to run; if None, it will run forever until error or KeyboardInterrupt
    loop_n :
        How many loops to run; if None, it will run forever until error or KeyboardInterrupt
        - if current loop is incomplete, it will be counted as the first loop for completion.
        - if both step_n and loop_n are provided, the process will stop as soon as either condition is met.
    competition :
    do_truncate :
        If set to True, the logger will truncate the future log messages by calling `logger.storage.truncate`.


    Auto R&D Evolving loop for models in a Kaggle scenario.
    You can continue running session by
    .. code-block:: bash
        dotenv run -- python rdagent/app/data_science/loop.py [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional parameter
        rdagent kaggle --competition playground-series-s4e8  # You are encouraged to use this one.
    """
    if competition is not None:
        DS_RD_SETTING.competition = competition

    if DS_RD_SETTING.competition:
        if DS_RD_SETTING.scen.endswith("KaggleScen"):
            download_data(competition=DS_RD_SETTING.competition, settings=DS_RD_SETTING)
        else:
            if not Path(f"{DS_RD_SETTING.local_data_path}/{competition}").exists():
                logger.error(f"Please prepare data for competition {competition} first.")
                return
    else:
        logger.error("Please specify competition name.")
    if path is None:
        kaggle_loop = DataScienceRDLoop(DS_RD_SETTING)
    else:
        kaggle_loop = DataScienceRDLoop.load(path, output_path, do_truncate)
    kaggle_loop.run(step_n=step_n, loop_n=loop_n)


if __name__ == "__main__":
    fire.Fire(main)
