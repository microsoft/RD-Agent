import re 
from rdagent.core.proposal import ExpGen
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend




class DS_EnsembleExpGen(ExpGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_response_schema = APIBackend().supports_response_schema()