import asyncio
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from rdagent.app.agentic_sys.conf import AgenticSysSetting
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError, PolicyError, RunnerError
from rdagent.core.proposal import Experiment2Feedback, ExperimentFeedback, ExpGen, Trace
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.agentic_sys.exp import AgenticSysExperiment
from rdagent.core.proposal import ExpGen


class AgenticSysRDLoop(RDLoop):
    # NOTE: we move the DataScienceRDLoop here to be easier to be imported
    # Maintain experiment loop history and context
    # support multi-iteration optimization
    skip_loop_error = (CoderError, RunnerError)
    withdraw_loop_error = (PolicyError,)

    def __init__(self, PROP_SETTING: AgenticSysSetting):

        scen = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)
        self.scen: Scenario = scen
        self.exp_gen: ExpGen = import_class(PROP_SETTING.exp_gen)(scen)

        self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
        self.runner: Developer = import_class(PROP_SETTING.runner)(scen)

        self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.feedback)(scen)
        self.trace = Trace(scen=scen)

        #Store configuration
        self.setting = PROP_SETTING

        super(RDLoop, self).__init__()

        logger.info(f"AgenticSysRDLoop initialized for competition: {PROP_SETTING.competition}")

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        exp = await self.exp_gen.async_gen(self.trace, self)
        return {"exp_gen": exp}

    def record(self, prev_out: dict[str, Any]):
        cur_loop_id = prev_out[self.LOOP_IDX_KEY]

        if (e := prev_out.get(self.EXCEPTION_KEY, None)) is None:
            exp = prev_out["running"]
            self.trace.sync_dag_parent_and_hist((exp, prev_out["feedback"]), cur_loop_id)
        else:
            exp: DSExperiment = prev_out["direct_exp_gen"] if isinstance(e, CoderError) else prev_out["coding"]
            self.trace.sync_dag_parent_and_hist(
                (
                    exp,
                    ExperimentFeedback.from_exception(e),
                ),
                cur_loop_id,
            )
