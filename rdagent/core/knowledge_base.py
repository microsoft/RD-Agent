from pathlib import Path

import dill as pickle  # type: ignore[import-untyped]

from rdagent.core.serialization import dump as secure_pickle_dump
from rdagent.core.serialization import load as secure_pickle_load
from rdagent.log import rdagent_logger as logger


class KnowledgeBase:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.load()

    def load(self) -> None:
        if self.path is not None and self.path.exists():
            with self.path.open("rb") as f:
                loaded = secure_pickle_load(f, serializer=pickle)
                if isinstance(loaded, dict):
                    self.__dict__.update({k: v for k, v in loaded.items() if k != "path"})
                else:
                    self.__dict__.update({k: v for k, v in loaded.__dict__.items() if k != "path"})

    def dump(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as f:
                secure_pickle_dump(self.__dict__, f, serializer=pickle)
        else:
            logger.warning("KnowledgeBase path is not set, dump failed.")
