import pandas as pd
from rdagent.components.coder.model_coder.model import ModelFBWorkspace
from rdagent.core.experiment import FBWorkspace
from rdagent.utils.env import DSDockerConf, DockerEnv, KGDockerConf, QlibDockerConf

from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent


def get_data_conf(init_val):
    # TODO: inherent from the benchmark base class
    import torch
    # TODO: design this step in the workflow
    in_dim = 1000
    in_channels = 128
    exec_config = {"model_eval_param_init": init_val}
    node_feature = torch.randn(in_dim, in_channels)
    edge_index = torch.randint(0, in_dim, (2, 2000))
    return (node_feature, edge_index), exec_config


class ModelImpValEval:
    """
    Evaluate the similarity of the model structure by changing the input and observe the output.

    Assumption:
    - If the model structure is similar, the output will change in similar way when we change the input.

    Challenge:
    - The key difference between it and implementing models is that we have parameters in the layers (Model operators often have no parameters or are given parameters).
    - we try to initialize the model param in similar value. So only the model structure is different.

    Comparing the correlation of following sequences
    - modelA[init1](input1).hidden_out1, modelA[init1](input2).hidden_out1, ...
    - modelB[init1](input1).hidden_out1, modelB[init1](input2).hidden_out1, ...

    For each hidden output, we can calculate a correlation. The average correlation will be the metrics.
    """

    def evaluate(self, gt: FBWorkspace, gen: FBWorkspace,task_name:str):
        env = DockerEnv(KGDockerConf())

        # __import__('ipdb').set_trace()
        # from IPython import embed; embed()
        gt.file_dict.keys()
        gen.file_dict.keys()

        test_ws = FBWorkspace()  # create a workspace for test
        test_ws.prepare()
        test_ws.inject_files(
            **{
                "gt_model.py": gt.file_dict["model.py"],
                "gen_model.py": gen.file_dict["model.py"],
                "test_model.py": f'MODEL_NAME = "{task_name}"\n' +(DIRNAME.parent / "eval_tests" / "model.py").open().read()
            }) # TODO: change init way by model name
        print (f'MODEL_NAME = "{task_name}"\n' +(DIRNAME.parent / "eval_tests" / "model.py").open().read())
        print ("start")
        
        test_ws.execute(env=env, entry="python test_model.py")
        print ("end")
        # print((test_ws.workspace_path / "result.csv").read_text())
        # print(type(res), res)
        res = pd.read_csv(test_ws.workspace_path / "result.csv", index_col=0).squeeze()
        return res
