# TODO: inherent from the benchmark base class
import torch

from rdagent.components.coder.model_coder.model import ModelFBWorkspace


def get_data_conf(init_val):
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

    def evaluate(self, gt: ModelFBWorkspace, gen: ModelFBWorkspace):
        round_n = 10

        eval_pairs: list[tuple] = []

        # run different input value
        for _ in range(round_n):
            # run different model initial parameters.
            for init_val in [-0.2, -0.1, 0.1, 0.2]:
                _, gt_res = gt.execute(input_value=init_val, param_init_value=init_val)
                _, res = gen.execute(input_value=init_val, param_init_value=init_val)
                eval_pairs.append((res, gt_res))

        # flat and concat the output
        res_batch, gt_res_batch = [], []
        for res, gt_res in eval_pairs:
            res_batch.append(res.reshape(-1))
            gt_res_batch.append(gt_res.reshape(-1))
        res_batch = torch.stack(res_batch)
        gt_res_batch = torch.stack(gt_res_batch)

        res_batch = res_batch.detach().numpy()
        gt_res_batch = gt_res_batch.detach().numpy()

        # pearson correlation of each hidden output
        def norm(x):
            return (x - x.mean(axis=0)) / x.std(axis=0)

        dim_corr = (norm(res_batch) * norm(gt_res_batch)).mean(axis=0)  # the correlation of each hidden output

        # aggregate all the correlation
        avr_corr = dim_corr.mean()
        # FIXME:
        # It is too high(e.g. 0.944) .
        # Check if it is not a good evaluation!!
        # Maybe all the same initial params will results in extreamly high correlation without regard to the model structure.
        return avr_corr
