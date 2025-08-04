from typing import Generic, Optional, NamedTuple, Callable, Union
from typing import Generic, TypeVar, Protocol
from abc import ABC, abstractmethod

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]
Args = TypeVar("Args")

class MCTSConfig(NamedTuple):
    output_trace_in_each_iter: bool = True
    w_exp: float = 1.
    depth_limit: int = 10
    breadth_limit: int = 3
    n_iters: int = 20
    simulate_strategy: str | Callable[[list[float]], int] = 'max'
    disable_tqdm: bool = True
    temperature: float = 0.0
    temperature_decay_ratio: float = 0.75
    gamma: float = 1.0
    add_kl: bool = False
    consider_diversity: bool = True
    length_penalty: float = 1.25

class HyPoSearchArgs(NamedTuple):
    n_actions: int = 16
    n_init_actions: int = 16
    reward_alpha: float = 0.5
    reward_confidence_default: float = 0.8
    depth_limit: int = 32
    force_terminating_on_depth_limit: bool = False
    breadth_limit: int = 16
    similarity_threshold: float = .99
    negative_gen: bool = False
    kl_coeff: float = 0.02
    disable_tqdm: bool = True
    no_self_eval: bool = False
    use_code: bool = False
    use_mcq: bool = False
    eval_mode: bool = False
    init_temperature: float = 1.0
    temperature: float = 1.0
    get_tp_zero: bool = False
    model_type: str = 'mistral'
    include_gt: bool = True
    verbose: bool = False



class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> State: ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_example(self, example: Example) -> None:
        self.example = example


class SearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...
    
    @abstractmethod
    def get_values(self, state: State, action: Action) -> list[tuple[float, bool]]: ...

    def update_example(self, example: Example) -> None:
        self.example = example


class HasTerminalStateAndTrace(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> HasTerminalStateAndTrace: ...


