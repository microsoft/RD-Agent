"""Utilities shared by benchmark evaluators."""

from __future__ import annotations

import importlib
import logging
import re
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)


def _guess_dataset_var(mod_path: str) -> str:
    """Guess the dataset variable name from an OpenCompass module path.

    Convention: ``opencompass.configs.datasets.<name>.<name>_gen_<hash>``
    exports ``<name>_datasets``.  E.g.:
      - ``bbh.bbh_gen_ee62e9``       → ``bbh_datasets``
      - ``gsm8k.gsm8k_gen_1d7fe4``   → ``gsm8k_datasets``
      - ``ARC_c.ARC_c_gen_1e0de5``   → ``ARC_c_datasets``
    """
    # Take the parent package name (e.g. "bbh" from "...datasets.bbh.bbh_gen_xxx")
    parts = mod_path.rsplit(".", 2)
    if len(parts) >= 2:
        parent = parts[-2]  # e.g. "bbh", "gsm8k", "ARC_c"
        return f"{parent}_datasets"
    return "datasets"


def build_dataset_imports_explicit(dataset_imports: str | Iterable[str]) -> List[Dict[str, object]]:
    """Build explicit dataset import specs for the OpenCompass config template.

    Resolve explicit dataset variable names to avoid `import *`, which leaks
    non-serializable objects (e.g. `os`, `f` from BBH) and breaks mmengine's
    config dump+reload in the CLI.

    The returned structure matches `opencompass_template.yaml` expectation:
    `[{ "module": "...", "names": ["datasets", "..._datasets"] }, ...]`.
    """
    modules = [dataset_imports] if isinstance(dataset_imports, str) else list(dataset_imports)
    explicit: List[Dict[str, object]] = []
    for mod_path in modules:
        try:
            mod = importlib.import_module(mod_path)
            names = [
                attr
                for attr in dir(mod)
                if (attr == "datasets" or attr.endswith("_datasets")) and isinstance(getattr(mod, attr), list)
            ]
            if not names:
                guessed = _guess_dataset_var(mod_path)
                logger.warning(
                    "No dataset variables found in %s, guessing '%s'",
                    mod_path, guessed,
                )
                names = [guessed]
            explicit.append({"module": mod_path, "names": names})
        except Exception as e:
            guessed = _guess_dataset_var(mod_path)
            logger.warning(
                "Failed to import %s for explicit name resolution: %s. "
                "Guessing variable name '%s'.",
                mod_path, e, guessed,
            )
            explicit.append({"module": mod_path, "names": [guessed]})
    return explicit
