from pathlib import Path
from typing import Literal

from pydantic_settings import SettingsConfigDict

from rdagent.app.kaggle.conf import KaggleBasePropSetting


class AgenticSysSetting(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="AS_", protected_namespaces=())

    # Main components
    ## Scen
    scen: str = "rdagent.scenarios.data_science.scen.KaggleScen"
    """
    Scenario class for data science tasks.
    - For Kaggle competitions, use: "rdagent.scenarios.data_science.scen.KaggleScen"
    - For custom data science scenarios, use: "rdagent.scenarios.data_science.scen.DataScienceScen"
    """
    hypothesis_gen: str = "rdagent.scenarios.data_science.proposal.exp_gen.router.ParallelMultiTraceExpGen"
    interactor: str = "rdagent.components.interactor.SkipInteractor"
    trace_scheduler: str = "rdagent.scenarios.data_science.proposal.exp_gen.trace_scheduler.RoundRobinScheduler"
    """Hypothesis generation class"""

    summarizer: str = "rdagent.scenarios.data_science.dev.feedback.DSExperiment2Feedback"


DS_RD_SETTING = DataScienceBasePropSetting()
