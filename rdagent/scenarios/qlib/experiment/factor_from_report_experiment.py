from copy import deepcopy

from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.utils.agent.tpl import T


class QlibFactorFromReportScenario(QlibFactorScenario):
    def __init__(self) -> None:
        super().__init__()
        self._rich_style_description = deepcopy(T(".prompts:qlib_factor_from_report_rich_style_description").r())

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description
