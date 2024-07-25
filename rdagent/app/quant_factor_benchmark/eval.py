import os
from pathlib import Path
import pickle
import time
from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import (
    FactorTestCaseLoaderFromJsonFile,
)

from rdagent.components.benchmark.conf import BenchmarkSettings
from rdagent.components.benchmark.eval_method import FactorImplementEval
from rdagent.core.utils import import_class

from rdagent.core.utils import import_class
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario

from pprint import pprint

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

# 6.save the result
logger.log_object(res)
