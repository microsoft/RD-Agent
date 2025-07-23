from rdagent.components.workflow.conf import BasePropSetting


class FINETUNE_PROP_SETTING(BasePropSetting):
    """Basic PROP_SETTING skeleton for the finetune scenario.

    You can later fill the concrete class paths for hypothesis generator, coder, runner, etc.
    """

    # Scenario class path
    scen: str = "rdagent.scenarios.finetune.scenario.FinetuneScenario"

    # Place-holders for core components – please replace with real classes later.
    hypothesis_gen: str = ""
    hypothesis2experiment: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""
