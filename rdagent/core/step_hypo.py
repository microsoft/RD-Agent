from mcts_config import SearchConfig, WorldModel, SearchAlgorithm,HyPoSearchArgs
from typing import Generic, TypeVar, Protocol
from rdagent.core.experiment import Experiment
from rdagent.core.evaluation import Feedback
from proposal import ExperimentFeedback, MCTSNode

class StepHypoConfig(SearchConfig):
    def __init__(self, args: HyPoSearchArgs) -> None:
        super().__init__()
        self.example = None
        self.double_actions = False
        self.n_actions = args.n_actions
        self.n_init_actions = args.n_init_actions
        self.force_terminating_on_depth_limit = args.force_terminating_on_depth_limit
        self.depth_limit = args.depth_limit
        self.reward_alpha = args.reward_alpha
        self.reward_confidence_default = args.reward_confidence_default
        self.action_size = args.breadth_limit
        self.similarity_threshold = args.similarity_threshold
        self.negative_gen = args.negative_gen
        
        self.ref_policy_model = args.ref_policy_model
        
        self.base_tokenizer = args.base_tokenizer
        self.generation_config = args.generation_config        
        self.kl_coeff = args.kl_coeff
        self.disable_tqdm = args.disable_tqdm
        
        self.no_self_eval = args.no_self_eval
        self.reward_model = args.reward_model
        self.reward_tokenizer = args.reward_tokenizer
        
        self.use_code = args.use_code
        self.use_mcq = args.use_mcq
        self.eval_mode = args.eval_mode
        
        self.init_temperature = args.init_temperature
        self.temperature = args.temperature

    def get_actions(self, policy_model, state: tuple[Experiment, ExperimentFeedback, MCTSNode] , add_kl: bool = False) -> list[StepLMAction]:
        
        return 
    

    def get_values(self, state, action):
        return 

