"""
Differences from kaggle loop
- focused on a specific component(we must simplify it to align the grainularity of the idea and experiment).
- replace the idea proposal to another component.
- (trick) we don't want to develop again. we want to reused the code in the BO-process
    - cached Developer(input is same idea, return the cached solution)
        - the cache can be disabled.
    - evaluation results.


- Align the nouns:
    - e: Workspace
    - h: tasks or hypothesis?
"""


from rdagent.core.developer import Developer
from rdagent.core.experiment import Workspace
from rdagent.core.proposal import HypothesisGen


class BODev(Developer):
    """
    Differences:
    - save <h, e, s> results.
    - self evaluate a solution based <e, s>
    - directly query previous  <h, e> based on e.
    """
    def __init__(self):
        self.hypo2exp  # 
        self.dev  # normal dev 
        ...  # knowledge storage
    
    def evaluate(self, ws: Workspace):
        ...

    def udpate_feedback(self,  e,s):
        ...


class BOHypothesisGen(HypothesisGen):

    def __init__(self, scen: Scenario, bodev: BODev) -> None:
        self.bodev = bodev
        super().__init__(scen)

    def gen(self, ...):
        # 1) exploration : propose idea
        ideas = ...
        # 2) evaluate ideas with self.bodev
        # ..... scors distribution
        # 3) sample idea based on ideas & scores(as weight)
        return selected_idea


# - interface:
# - implemenation: use RepoAnalyzer + key code => score


class BOLoop:
    @measure_time
    def __init__(self, PROP_SETTING: BasePropSetting):
        with logger.tag("init"):
            ...
            self.bodev = BODev
            self.bohypogen = BOHypothesisGen(...,  self.bodev)
            ...

    ...
    def running(self):  #  feedback
        # collect <h, e, s>
        # or feeback
        e, s = self.trace ...
        self.bodev.update_feedack(e, s)  # 
