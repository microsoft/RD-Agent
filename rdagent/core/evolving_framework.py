from __future__ import annotations
from ..app.utils.gpu_utils import setup_gpu, optimize_model_for_gpu
import torch
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from rdagent.core.evaluation import EvaluableObj
from rdagent.core.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from rdagent.core.evaluation import Feedback
    from rdagent.core.scenario import Scenario


class Knowledge:
    pass


class QueriedKnowledge:
    pass


class EvolvingKnowledgeBase(KnowledgeBase):
    @abstractmethod
    def query(
        self,
    ) -> QueriedKnowledge | None:
        raise NotImplementedError


class EvolvableSubjects(EvaluableObj):
    """The target object to be evolved"""

    def clone(self) -> EvolvableSubjects:
        return copy.deepcopy(self)


ASpecificEvolvableSubjects = TypeVar("ASpecificEvolvableSubjects", bound=EvolvableSubjects)


@dataclass
class EvoStep(Generic[ASpecificEvolvableSubjects]):
    """At a specific step,
    based on
    - previous trace
    - newly RAG knowledge `QueriedKnowledge`

    the EvolvableSubjects is evolved to a new one `EvolvableSubjects`.

    (optional) After evaluation, we get feedback `feedback`.
    """

    evolvable_subjects: ASpecificEvolvableSubjects

    queried_knowledge: QueriedKnowledge | None = None
    feedback: Feedback | None = None


class EvolvingStrategy(ABC, Generic[ASpecificEvolvableSubjects]):
    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def evolve(
        self,
        *evo: ASpecificEvolvableSubjects,
        evolving_trace: list[EvoStep[ASpecificEvolvableSubjects]] | None = None,
        queried_knowledge: QueriedKnowledge | None = None,
        **kwargs: Any,
    ) -> ASpecificEvolvableSubjects:
        """The evolving trace is a list of (evolvable_subjects, feedback) ordered
        according to the time.

        The reason why the parameter is important for the evolving.
        - evolving_trace: the historical feedback is important.
        - queried_knowledge: queried knowledge
        """

class GPUEnhancedEvolvingFramework:
    def __init__(self):
        self.device = setup_gpu()
        self.gpu_available = torch.cuda.is_available()
        
    def evolve_time_series_model(self, model_config, training_data):
        """Enhanced model evolution with GPU support"""
        
        # Create model
        model = self.create_model(model_config)
        
        if self.gpu_available:
            model = optimize_model_for_gpu(model)
            training_data = self.prepare_gpu_data_pipeline(training_data)
            
        # Training logic with GPU support
        return self.train_with_gpu(model, training_data)
    
    def prepare_gpu_data_pipeline(self, dataset):
        """Prepare data pipeline for GPU optimization"""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            dataset,
            batch_size=64 if self.gpu_available else 32,
            shuffle=True,
            num_workers=4 if self.gpu_available else 2,
            pin_memory=True if self.gpu_available else False
        )
    
    def train_with_gpu(self, model, data_loader):
        """Training procedure with GPU optimization"""
        model.train()
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Move to GPU
            data = data.to(self.device)
            target = target.to(self.device)
        return model
    

class RAGStrategy(ABC, Generic[ASpecificEvolvableSubjects]):
    """Retrieval Augmentation Generation Strategy"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.knowledgebase: EvolvingKnowledgeBase = self.load_or_init_knowledge_base(*args, **kwargs)

    @abstractmethod
    def load_or_init_knowledge_base(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> EvolvingKnowledgeBase:
        pass

    @abstractmethod
    def query(
        self,
        evo: ASpecificEvolvableSubjects,
        evolving_trace: list[EvoStep],
        **kwargs: Any,
    ) -> QueriedKnowledge | None:
        pass

    @abstractmethod
    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep[ASpecificEvolvableSubjects]],
        *,
        return_knowledge: bool = False,
        **kwargs: Any,
    ) -> Knowledge | None:
        """Generating new knowledge based on the evolving trace.
        - It is encouraged to query related knowledge before generating new knowledge.

        RAGStrategy should maintain the new knowledge all by itself.
        """

    @abstractmethod
    def dump_knowledge_base(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load_dumped_knowledge_base(self, *args: Any, **kwargs: Any) -> None:
        """This is to load the dumped knowledge base.
        It's mainly used in parallel coding of which several coder shares the same knowledge base.
        Then the agent should load the knowledge base from others before updating it.
        """
