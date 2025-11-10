
from rdagent.core.experiment import Task
from rdagent.core.proposal import ExpGen, Trace
from rdagent.scenarios.agentic_sys.exp import AgenticSysExperiment


class AgenticSysExpGen(ExpGen):
    def gen(self, trace: Trace) -> AgenticSysExperiment:
        dummy_task = Task("You should design a agentic system task")
        experiment = AgenticSysExperiment(sub_tasks=[dummy_task])
        return experiment
        
