from pydantic_settings import BaseSettings



class PropSetting(BaseSettings):
    """"""

    scen: str = "rdagent.scenarios.qlib.factor_proposal.QlibFactorScenario" 
    hypothesis_gen: str = "rdagent.components.idea_proposal.factor_proposal.FactorHypothesisGen"
    hypothesis2experiment: str = "rdagent.components.idea_proposal.factor_proposal.FactorHypothesis2Experiment"

    evolving_n: int = 10


PROP_SETTING = PropSetting()
