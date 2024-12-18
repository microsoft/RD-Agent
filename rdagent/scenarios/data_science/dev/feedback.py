import json
from pathlib import Path

import pandas as pd

from rdagent.components.knowledge_management.graph import UndirectedNode
from rdagent.core.experiment import Experiment
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis,
    HypothesisFeedback,
    Trace,
)
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import KG_SELECT_MAPPING
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment

from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace

class DSExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: DSExperiment, trace: DSTrace) -> HypothesisFeedback:
        return super().generate_feedback(exp, trace)