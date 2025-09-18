from rdagent.core.experiment import ASpecificExp
from rdagent.core.interactor import Interactor
from rdagent.core.proposal import Trace


class SkipInteractor(Interactor[ASpecificExp]):

    def interact(self, exp: ASpecificExp, trace: Trace) -> ASpecificExp:
        """
        Interact with the user to get feedback or confirmation.

        Responsibilities:
        - Present the current state of the experiment to the user.
        - Collect user input to guide the next steps in the experiment.
        - Rewrite the experiment based on user feedback.
        """
        return exp
