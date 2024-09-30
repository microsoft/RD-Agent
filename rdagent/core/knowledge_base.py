from pathlib import Path

import dill as pickle  # type: ignore[import-untyped]

from rdagent.log import rdagent_logger as logger


class KnowledgeBase:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.load()

    def load(self) -> None:
        if self.path is not None and self.path.exists():
            with self.path.open("rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    self.__dict__.update(loaded)
                else:
                    self.__dict__.update(loaded.__dict__)

    def dump(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.__dict__, self.path.open("wb"))
        else:
            logger.warning("KnowledgeBase path is not set, dump failed.")
