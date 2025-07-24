from datetime import timedelta

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import RD_Agent_TIMER_wrapper
from rdagent.core.proposal import ExperimentPlan, ExpPlanner
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace


class DSExperimentPlan(ExperimentPlan):
    """
    A specific plan for data science experiments.
    This plan can include various stages such as proposal, draft, and merge.
    """

    def __init__(self):
        super().__init__()
        self.setdefault("exp_gen", {}).setdefault("draft", False)
        self.setdefault("exp_gen", {}).setdefault("suggest_model_architecture", False)
        self.setdefault("exp_gen", {}).setdefault("suggest_model_ensemble", False)


class DSExpPlannerHandCraft(ExpPlanner[DSExperimentPlan]):
    """
    A specific planner for data science experiments.
    """

    def plan(self, trace: DSTrace) -> DSExperimentPlan:
        """
        Generate a plan for the experiment based on the trace.
        The plan should be a dictionary that contains the plan to each stage.
        trace is well selected into sub trace mode
        """
        plan = DSExperimentPlan()
        timer = RD_Agent_TIMER_wrapper.timer
        remain_percent = timer.remain_time() / timer.all_duration if timer.started else 1.0

        if not trace.sota_experiment():
            plan["exp_gen"]["draft"] = True
        elif trace.sota_experiment() and remain_percent > DS_RD_SETTING.model_architecture_suggestion_time_percent:
            plan["exp_gen"]["suggest_model_architecture"] = True
        # elif DS_RD_SETTING.merge_hours > 0:
        #     merge_percent = timedelta(hours=DS_RD_SETTING.merge_hours) / timer.all_duration
        #     if merge_percent < remain_percent < merge_percent + 0.1:
        #         plan["exp_gen"]["suggest_model_ensemble"] = True
        return plan
