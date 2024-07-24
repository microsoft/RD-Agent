from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from rdagent.core.exception import RunnerException
from rdagent.components.coder.factor_coder.CoSTEER.evaluators import (
    FactorCorrelationEvaluator,
    FactorEqualValueCountEvaluator,
    FactorEvaluator,
    FactorIndexEvaluator,
    FactorMissingValuesEvaluator,
    FactorOutputFormatEvaluator,
    FactorRowCountEvaluator,
    FactorSingleColumnEvaluator,
)


"""
Define EVAL_RES_ONLINE with an example
{
    <factor_name: str>: [
        (
            <FactorEvaluator: object>,
            if successfully run(call `.execute()`):
                [
                    (Evaluator,
                        if successfully evaluate it:
                            (feedback, metric),
                        else:
                            EvaluationException
                    ),
                    ... other evaluators ...
                ]
            else:
                <Run Exception>
        )
        ... more similar tuples ...
    ]
}
"""
EVAL_RES_ONLINE = Dict[
    str,
    List[Tuple[FactorEvaluator, Union[object, RunnerException]]],
]

def summarize_res(res: EVAL_RES_ONLINE) -> pd.DataFrame:
    # None: indicate that it raises exception and get no results
    sum_res = {}
    for factor_name, runs in res.items():
        for fi, err_or_res_l in runs:
            # NOTE:  str(fi) may not be unique!!  Because the workspace can be skipped when hitting the cache.
            uniq_key = f"{str(fi)},{id(fi)}"

            key = (factor_name, uniq_key)
            val = {}
            if isinstance(err_or_res_l, Exception):
                val["run factor error"] = str(err_or_res_l.__class__)
            else:
                val["run factor error"] = None
                for ev_obj, err_or_res in err_or_res_l:
                    if isinstance(err_or_res, Exception):
                        val[str(ev_obj)] = None
                    else:
                        feedback, metric = err_or_res
                        val[str(ev_obj)] = metric
            sum_res[key] = val

    return pd.DataFrame(sum_res)
