from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import (
    FactorTestCaseLoaderFromJsonFile,
)

from rdagent.components.benchmark.conf import BenchmarkSettings
from rdagent.components.benchmark.eval_method import FactorImplementEval
from rdagent.core.utils import import_class

from rdagent.core.utils import import_class
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario


# 1.read the settings
bs = BenchmarkSettings()

# 2.read and prepare the eval_data
test_cases = FactorTestCaseLoaderFromJsonFile().load(bs.bench_data_path)

# 3.declare the method to be tested and pass the arguments.

scen: Scenario = import_class(PROP_SETTING.factor_scen)()
generate_method = import_class(bs.bench_method_cls)(scen=scen)
 
# 4.declare the eval method and pass the arguments.
eval_method = FactorImplementEval(
    method=generate_method,
    test_cases=test_cases,
    scen=scen,
    catch_eval_except=True,
    test_round=bs.bench_test_round,
)

# 5.run the eval
res = eval_method.eval()

# TODO:
# - Run it:
# - factor input data generator;
#   - f_{gt}(input) => value_{gt}
#   - f_{llm}(input) => value_{llm}
#   - we have legal issue to release Input
# - Eval result:
#   -  check https://github.com/peteryang1/fincov2/blob/master/src/scripts/benchmark/analysis.py
