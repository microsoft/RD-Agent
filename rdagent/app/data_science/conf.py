from pydantic_settings import SettingsConfigDict

from rdagent.app.kaggle.conf import KaggleBasePropSetting


class DataScienceBasePropSetting(KaggleBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="DS_", protected_namespaces=())

    # Main components
    ## Scen
    scen: str = "rdagent.scenarios.data_science.scen.KaggleScen"
    """Scenario class for data mining model"""

    ## Workflow Related
    consecutive_errors: int = 5

    debug_timeout: int = 600
    """The timeout limit for running on debugging data"""
    full_timeout: int = 3600
    """The timeout limit for running on full data"""

    ### specific feature

    #### enable specification
    spec_enabled: bool = True

    proposal_version: str = "v1"
    coder_on_whole_pipeline: bool = False

    coder_max_loop: int = 10
    runner_max_loop: int = 3


DS_RD_SETTING = DataScienceBasePropSetting()
