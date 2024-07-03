from pydantic_settings import BaseSettings


class PropSetting(BaseSettings):
    """"""

    scen: str = "rdagent.scenarios.qlib.factor_proposal.QlibFactorScenario"  # a.b.c:XXXClass
    # TODO: inital keywards should be included in the settings
    ...


PROP_SETTING = PropSetting()
