from rdagent.components.coder.data_science.pipeline import PipelineCoSTEER
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


class FinetuneRDLoop(DataScienceRDLoop):
    """Minimal LLM finetune loop for early-stage single-run development.

    - Runs only once, no loop or history memory
    - Only supports Pipeline tasks
    - Trace, SOTA selection, and feedback are removed
    """

    def __init__(self, PROP_SETTING):
        logger.log_object(PROP_SETTING.task, tag="task")

        # Basic scenario setup
        scen = import_class(PROP_SETTING.scen)()
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")

        # Core components: experiment generator, coder, runner
        self.exp_gen = import_class(PROP_SETTING.hypothesis_gen)(scen)
        self.pipeline_coder = PipelineCoSTEER(scen)
        self.runner = DSCoSTEERRunner(scen)

        # Initialize loop base
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

    async def direct_exp_gen(self, prev_out: dict):
        """Experiment generation - single run, no history dependency"""
        exp = await self.exp_gen.async_gen(None, self)
        logger.log_object(exp)
        return exp

    def feedback(self, prev_out: dict):
        """Skip feedback phase, directly return simple success feedback"""
        logger.info("Skipping feedback generation for single-run finetune")
        # Directly pass the running result to the record phase
        return prev_out.get("running")

    def record(self, prev_out: dict):
        """Simple record of experiment result"""
        exp: DSExperiment = prev_out.get("feedback") or prev_out.get("running")

        if exp:
            logger.log_object(exp, tag="experiment_completed")
            logger.info("Finetune experiment completed successfully")
        else:
            logger.info("Finetune experiment failed or not executed")

        return exp
