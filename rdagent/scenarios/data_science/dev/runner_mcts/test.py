import unittest
from rdagent.scenarios.data_science.dev.runner_mcts import (
    DSRunnerMCTSMultiProcessEvolvingStrategy,
    MCTSNode,
)
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.CoSTEER.evolvable_subjects import FBWorkspace
from rdagent.components.coder.CoSTEER.evolving_strategy import CoSTEERQueriedKnowledge
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback

class DummyWorkspace(FBWorkspace):
    def __init__(self):
        self.file_dict = {"main.py":  """\
# 超参数设置
hyperparameters = {
    "learning_rate": 0.01,  # 初始学习率，可调
    "max_iter": 100,        # 迭代次数，可调
    "fit_intercept": True    # 是否拟合截距，可调
}

print("Hello World!")
print("Current hyperparameters:", hyperparameters)

# 生成简单的数据集
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# 初始化并训练线性模型
model = SGDRegressor(
    learning_rate='constant',
    eta0=hyperparameters["learning_rate"],
    max_iter=hyperparameters["max_iter"],
    fit_intercept=hyperparameters["fit_intercept"],
    random_state=42
)
model.fit(X, y)

# 输出模型系数
print("Trained model coefficients:", model.coef_)
"""}
        self.change_summary = ""
        self.workspace_path = "/tmp"
    @property
    def all_codes(self):
        return  """\
# 超参数设置
hyperparameters = {
    "learning_rate": 0.01,  # 初始学习率，可调
    "max_iter": 100,        # 迭代次数，可调
    "fit_intercept": True    # 是否拟合截距，可调
}

print("Hello World!")
print("Current hyperparameters:", hyperparameters)

# 生成简单的数据集
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# 初始化并训练线性模型
model = SGDRegressor(
    learning_rate='constant',
    eta0=hyperparameters["learning_rate"],
    max_iter=hyperparameters["max_iter"],
    fit_intercept=hyperparameters["fit_intercept"],
    random_state=42
)
model.fit(X, y)

# 输出模型系数
print("Trained model coefficients:", model.coef_)
"""
    def inject_files(self, **kwargs):
        self.file_dict.update(kwargs)

class DummyTask(CoSTEERTask):
    def __init__(self):
        super().__init__(name="dummy_task")
    def get_task_information(self):
        return "Dummy task"

class DummyFeedback(CoSTEERSingleFeedback):
    def __init__(self):
        super().__init__(
            execution="pass",
            return_checking="pass",
            code="print('hello')"
        )
        self.acceptable = True
        self.score = 1.0
        self.hyperparameter_tuning_suggestion = None

class DummyQueriedKnowledge(CoSTEERQueriedKnowledge):
    def __init__(self):
        self.task_to_former_failed_traces = {"Dummy task": [[]]}

from rdagent.log.storage import FileStorage
from pathlib import Path
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment

from rdagent.log.utils import extract_loopid_func_name
log_path = Path("/home/bowen/workspace/JobAndExp/amlt_project/amlt/coherent-macaque/combined_logs/aerial-cactus-identification.1")

traces = []
for msg in FileStorage(log_path).iter_msg(tag="trace"):
    loop_id, fn = extract_loopid_func_name(msg.tag)
    traces.append(msg.content)
Tr = traces[-5]



class DummyScenario:
    initial_workspace = DummyWorkspace()
    metric_direction = True
    gt_workspace = DummyWorkspace()
    competition = "tabular-playground-series-dec-2021"



class DummySettings:
    diff_mode = False

class TestMCTSStrategy(unittest.TestCase):
    def test_mcts_flow(self):
        scen =  Tr.scen#DummyScenario()
        settings = DummySettings()
        strategy = DSRunnerMCTSMultiProcessEvolvingStrategy(scen, settings, max_iterations=1)
        task = DummyTask()
        workspace = DummyWorkspace()
        feedback = DummyFeedback()
        queried_knowledge = DummyQueriedKnowledge()
        result = strategy.implement_one_task(
            target_task=task,
            queried_knowledge=queried_knowledge,
            workspace=workspace,
            prev_task_feedback=feedback,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("main.py", result)

if __name__ == "__main__":
    unittest.main()



