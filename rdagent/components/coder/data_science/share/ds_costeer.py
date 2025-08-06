from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback


class DSCoSTEER(CoSTEER):
    def get_develop_max_seconds(self) -> int | None:
        """
        The coder uses the scenario's real debug timeout as the maximum seconds for development.
        """
        return int(self.scen.real_debug_timeout() * self.settings.max_seconds_multiplier)

    def compare_and_pick_fb(self, base_fb: CoSTEERMultiFeedback | None, new_fb: CoSTEERMultiFeedback | None) -> bool:
        # In data science, we only have a single feedback
        base_fb = base_fb[0]
        new_fb = new_fb[0]

        def compare_scores(s1, s2) -> bool:
            if s1 is None:
                if s2 is None:  # FIXME: is this possible?
                    return False
                if s2 is not None:
                    return True
            else:
                if s2 is None:  # new is invalid
                    return False
                else:
                    return (s2 > s1) == self.scen.metric_direction

        return compare_scores(base_fb.score, new_fb.score)
