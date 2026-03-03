"""Utilities shared by benchmark evaluators."""

from __future__ import annotations

import importlib
from typing import Dict, Iterable, List


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
            explicit.append({"module": mod_path, "names": names})
        except Exception:
            explicit.append({"module": mod_path, "names": []})
    return explicit
