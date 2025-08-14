from rdagent.components.coder.data_science.pipeline import PipelineCoSTEER
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace


class FinetuneRDLoop(DataScienceRDLoop):
    """A minimal RD loop tailored for LLM fine-tuning.

    - Uses the LLM finetune scenario specified by PROP_SETTING.scen
    - Focuses on Pipeline coder only
    - Reuses DS runner and feedback
    """

    def __init__(self, PROP_SETTING):
        logger.log_object(PROP_SETTING.task, tag="task")

        scen = import_class(PROP_SETTING.scen)()
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")

        # exp generation
        self.ckp_selector = import_class(PROP_SETTING.selector_name)()
        self.sota_exp_selector = import_class(PROP_SETTING.sota_exp_selector_name)()
        self.exp_gen = import_class(PROP_SETTING.hypothesis_gen)(scen)

        # minimal coder set: only Pipeline
        self.pipeline_coder = PipelineCoSTEER(scen)

        # runner & summarizer
        self.runner = DSCoSTEERRunner(scen)
        self.summarizer = DSExperiment2Feedback(scen)

        # trace
        self.trace = DSTrace(scen=scen)

        # initialize loop base
        super(RDLoop, self).__init__()

    def coding(self, prev_out: dict):
        exp = prev_out["direct_exp_gen"]
        for tasks in exp.pending_tasks_list:
            exp.sub_tasks = tasks
            with logger.tag(f"{exp.sub_tasks[0].__class__.__name__}"):
                # Only support Pipeline task in LLM finetune
                if isinstance(exp.sub_tasks[0], PipelineTask):
                    exp = self.pipeline_coder.develop(exp)
                else:
                    # Fallback: treat all tasks as pipeline for simplicity
                    exp = self.pipeline_coder.develop(exp)
            exp.sub_tasks = []
        logger.log_object(exp)
        return exp

    def running(self, prev_out: dict):
        exp: DSExperiment = prev_out["coding"]
        if exp.is_ready_to_run():
            new_exp = self.runner.develop(exp)
            logger.log_object(new_exp)
            exp = new_exp
        return exp
