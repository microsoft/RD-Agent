from __future__ import annotations

from rdagent.core.evolving_framework import KnowledgeBase
from rdagent.core.proposal import Trace

RLTrace = Trace["RLPostTrainingScen", KnowledgeBase]
