
from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class AgenticSysSetting(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="AS_", protected_namespaces=())

    competition: str | None = None

    # Main components
    ## Scen
    scen: str = "rdagent.scenarios.agentic_sys.scen.AgenticSysScen"
    """
    Scenario class for data science tasks.
    - For Kaggle competitions, use: "rdagent.scenarios.data_science.scen.KaggleScen"
    - For custom data science scenarios, use: "rdagent.scenarios.data_science.scen.DataScienceScen"
    """
    exp_gen: str = "rdagent.scenarios.agentic_sys.proposal.AgenticSysExpGen"
    coder: str = "rdagent.scenarios.agentic_sys.dev.AgenticSysCoder"
    runner: str = "rdagent.scenarios.agentic_sys.dev.AgenticSysRunner"

    feedback: str = "rdagent.scenarios.agentic_sys.feedback.AgenticSysExp2Feedback"


ASYS_RD_SETTING = AgenticSysSetting()
