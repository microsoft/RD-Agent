from rdagent.app.kaggle.conf import KaggleBasePropSetting
from rdagent.core.conf import ExtendedSettingsConfigDict


class DataScienceBasePropSetting(KaggleBasePropSetting):
    model_config = ExtendedSettingsConfigDict(env_prefix="DS_", protected_namespaces=())

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

    # BO related

    idea_bo_mode: bool = True
    """Whether to use BO mode for idea generation (with given component, just for idea generation)"""

    component_bo_mode: bool = True
    """Whether to use BO mode for component selection ([component,idea] pair evaluation)"""

    batch_bo_eval: bool = False
    """Whether to use batch evaluation in BO (i.e. evaluate multiple proposals at once)"""

    idea_bo_step: int = 5
    """The step size for idea generation BO"""

    component_bo_step: int = 5
    """The step size for component selection BO"""



DS_RD_SETTING = DataScienceBasePropSetting()
