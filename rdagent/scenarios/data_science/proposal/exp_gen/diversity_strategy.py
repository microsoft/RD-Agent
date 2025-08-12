from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace


class DiversityContextStrategy(ABC):
    """
    An abstract base class for strategies that determine when to inject
    cross-trace diversity context into the generation process.
    """

    @abstractmethod
    def should_inject(self, trace: DSTrace, local_selection: tuple[int, ...]) -> bool:
        """
        Decides whether to inject diversity context based on the current state of the trace
        and the selection for the next experiment.

        Args:
            trace: The full DSTrace object.
            local_selection: The parent node selection for the new experiment.

        Returns:
            True if context should be injected, False otherwise.
        """
        raise NotImplementedError


class InjectAtRootStrategy(DiversityContextStrategy):
    """
    A strategy that injects diversity context only when creating a new root for a sub-trace.
    """

    def should_inject(self, trace: DSTrace, local_selection: tuple[int, ...]) -> bool:
        """Injects only when `local_selection` indicates a new trace root."""
        return local_selection == trace.NEW_ROOT


class InjectUntilSOTAGainedStrategy(DiversityContextStrategy):
    """
    A strategy that injects diversity context until the first SOTA (State-of-the-Art)
    experiment is achieved within the current sub-trace.
    """

    def should_inject(self, trace: DSTrace, local_selection: tuple[int, ...]) -> bool:
        """
        Injects if the sub-trace corresponding to the `local_selection` has not
        yet produced a successful SOTA experiment.
        """
        # If starting a new trace, there's no SOTA yet, so inject.
        if local_selection == trace.NEW_ROOT:
            return True

        # Check for SOTA within the specific sub-trace.
        return trace.sota_experiment(selection=local_selection) is None


class AlwaysInjectStrategy(DiversityContextStrategy):
    """
    A strategy that always injects diversity context.
    """

    def should_inject(self, trace: DSTrace, local_selection: tuple[int, ...]) -> bool:
        """Always returns True to indicate that context should be injected."""
        return True
