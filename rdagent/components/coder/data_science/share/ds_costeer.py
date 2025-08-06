from rdagent.components.coder.CoSTEER import CoSTEER


class DSCoSTEER(CoSTEER):
    def get_develop_max_seconds(self) -> int | None:
        """
        The coder uses the scenario's real debug timeout as the maximum seconds for development.
        """
        return int(self.scen.real_debug_timeout() * self.settings.max_seconds_multiplier)
