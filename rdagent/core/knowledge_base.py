from pathlib import Path
import dill as pickle

from rdagent.log import rdagent_logger as logger


class KnowledgeBase:
    def __init__(self, path: str | Path = None) -> None:
        self.path = Path(path) if path else None
        self.load()

    def load(self):
        if self.path is not None and self.path.exists():
            self.__dict__.update(
                pickle.load(open(self.path, "rb")).__dict__
            )  # TODO: because we need to align with init function, we need a less hacky way to do this

    def dump(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self, open(self.path, "wb"))
        else:
            logger.warning("KnowledgeBase path is not set, dump failed.")
