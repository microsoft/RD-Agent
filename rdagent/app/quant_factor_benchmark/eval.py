from rdagent.scenarios.qlib.factor_task_loader.json_loader import (
    FactorTestCaseLoaderFromJsonFile,
)

from rdagent.components.benchmark.conf import BenchmarkSettings
from rdagent.components.benchmark.eval_method import FactorImplementEval
from rdagent.core.utils import import_class

# 1.read the settings
bs = BenchmarkSettings()

# 2.read and prepare the eval_data
test_cases = FactorTestCaseLoaderFromJsonFile().load(bs.bench_data_path)

# 3.declare the method to be tested and pass the arguments.

method_cls = import_class(bs.bench_method_cls)
generate_method = method_cls()

# 4.declare the eval method and pass the arguments.
eval_method = FactorImplementEval(
    method=generate_method,
    test_cases=test_cases,
    catch_eval_except=True,
    test_round=bs.bench_test_round,
)

# 5.run the eval
res = eval_method.eval()
