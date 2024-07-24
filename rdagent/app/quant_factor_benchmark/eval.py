import os
from pathlib import Path
import pickle
import time
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
pprint(res)

res_workspace = (Path().cwd() / "git_ignore_folder" / "eval_results").absolute()
print(str(res_workspace))

# Save results
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))

if not os.path.exists(str(res_workspace)):
    os.makedirs(str(res_workspace))

df_file_path = res_workspace / ("result_" + timestamp + ".csv")
res_pkl_path = res_workspace / ("res_promptV2" + timestamp + ".pkl")
res_pkl_path = res_workspace / ("res_promptV2" + timestamp + ".pkl")
with open(str(res_pkl_path), "wb") as file:
    # file.write(str(res))
    pickle.dump(res, file)

# TODO:
# - Run it:
# - factor input data generator;
#   - f_{gt}(input) => value_{gt}
#   - f_{llm}(input) => value_{llm}
#   - we have legal issue to release Input
# - Eval result:
#   -  check https://github.com/peteryang1/fincov2/blob/master/src/scripts/benchmark/analysis.py
