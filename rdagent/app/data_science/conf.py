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
    # - [ ] rdagent/components/coder/data_science/raw_data_loader/__init__.py: make spec implementation optional
    # - [ ] move spec responsibility into  rdagent/scenarios/data_science/share.yaml
    # - [ ] make all spec.md optional;  but replace it with the test & responsibility.   "spec/.*\.md".
    # - [ ] replace yaml render with target test.  "spec > .yaml data_science !out_spec !task_spec model_spec"
    # - [ ] At the head of all tests, emphasis the function to be tested.


DS_RD_SETTING = DataScienceBasePropSetting()
