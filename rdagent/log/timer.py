import re
from datetime import datetime, timedelta

from rdagent.core.utils import SingletonBaseClass
from rdagent.log import rdagent_logger as logger


class RDAgentTimer(SingletonBaseClass):
    def __init__(self) -> None:
        self.started = False
        self.target_time = None
        self.all_duration = None
        self.remain_time_duration = None

    def reset(self, all_duration: str | timedelta) -> None:
        if isinstance(all_duration, str):
            pattern = re.compile(r"^\s*(\d*\.?\d+)\s*([smhd]?)\s*$")

            match = pattern.match(all_duration)
            if not match:
                return None
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "s":
                self.all_duration = timedelta(seconds=value)
            elif unit == "m":
                self.all_duration = timedelta(minutes=value)
            elif unit == "h":
                self.all_duration = timedelta(hours=value)
            elif unit == "d":
                self.all_duration = timedelta(days=value)
            else:
                self.all_duration = timedelta(seconds=value)
        elif isinstance(all_duration, timedelta):
            self.all_duration = all_duration
        self.target_time = datetime.now() + self.all_duration
        logger.info(f"Timer set to {self.all_duration} seconds and counting down.")
        self.started = True
        return None
    
    def restart_by_remain_time(self) -> None:
        if self.remain_time_duration is not None:
            self.target_time = datetime.now() + self.remain_time_duration
            self.started = True
            logger.info(f"Timer restarted with remaining time: {self.remain_time_duration}")
        else:
            logger.warning("No remaining time to restart the timer.")
        return None

    def add_duration(self, duration: timedelta) -> None:
        if self.started:
            logger.info(f"Adding {duration} to the timer. Currently {self.remain_time()} remains.")
            self.target_time = self.target_time + duration
            self.update_remain_time()

    def is_timeout(self) -> bool:
        if self.started:
            self.update_remain_time()
            if datetime.now() > self.target_time:
                return True
        return False
    
    def update_remain_time(self) -> None:
        if self.started:
            self.remain_time_duration = self.target_time - datetime.now()
        return None

    def remain_time(self) -> timedelta | None:
        if self.started:
            self.update_remain_time()
            return self.remain_time_duration
        return None


RD_Agent_TIMER = RDAgentTimer()
