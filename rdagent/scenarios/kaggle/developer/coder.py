import json
from typing import Dict, List

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder import FactorCoSTEER
from rdagent.components.coder.model_coder import ModelCoSTEER
from rdagent.core.developer import Developer
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import (
    KG_SELECT_MAPPING,
    KGModelExperiment,
)

KGModelCoSTEER = ModelCoSTEER
KGFactorCoSTEER = FactorCoSTEER
from rdagent.utils.agent.tpl import T

DEFAULT_SELECTION_CODE = """
import pandas as pd
def select(X: pd.DataFrame) -> pd.DataFrame:
    \"""
    Select relevant features. To be used in fit & predict function.
    \"""
    if X.columns.nlevels == 1:
        return X
    {% if feature_index_list is not none %}
    X = X.loc[:, X.columns.levels[0][{{feature_index_list}}].tolist()]
    {% endif %}
    X.columns = ["_".join(str(i) for i in col).strip() for col in X.columns.values]
    return X
"""


class KGModelFeatureSelectionCoder(Developer[KGModelExperiment]):
    def develop(self, exp: KGModelExperiment) -> KGModelExperiment:
        target_model_type = exp.sub_tasks[0].model_type
        assert target_model_type in KG_SELECT_MAPPING
        if len(exp.experiment_workspace.data_description) == 1:
            code = (
                Environment(undefined=StrictUndefined)
                .from_string(DEFAULT_SELECTION_CODE)
                .render(feature_index_list=None)
            )
        else:
            system_prompt = T("scenarios.kaggle.prompts:model_feature_selection.system").r(
                scenario=exp.scen.get_scenario_all_desc(),
                model_type=exp.sub_tasks[0].model_type,
            )
            user_prompt = T("scenarios.kaggle.prompts:model_feature_selection.user").r(
                feature_groups=[desc[0] for desc in exp.experiment_workspace.data_description]
            )

            chosen_index = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                    json_target_type=Dict[str, List[int]],
                )
            ).get("Selected Group Index", [i + 1 for i in range(len(exp.experiment_workspace.data_description))])
            chosen_index_to_list_index = [i - 1 for i in chosen_index]

            code = (
                Environment(undefined=StrictUndefined)
                .from_string(DEFAULT_SELECTION_CODE)
                .render(feature_index_list=chosen_index_to_list_index)
            )
        exp.experiment_workspace.inject_files(**{KG_SELECT_MAPPING[target_model_type]: code})
        return exp
