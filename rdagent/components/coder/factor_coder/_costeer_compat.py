"""Backward-compatibility shim for the pre-refactor CoSTEER module path.

Traces pickled before the CoSTEER relocation reference
``rdagent.components.coder.factor_coder.CoSTEER.*`` (the old location).
After the refactor the package lives at ``rdagent.components.coder.CoSTEER.*``,
so loading those pickles in ``rdagent ui`` fails with::

    ModuleNotFoundError: No module named
    'rdagent.components.coder.factor_coder.CoSTEER'

— see #1331. Install a ``sys.meta_path`` finder that transparently redirects
imports under the old path to the matching submodule under the new one, so
the pre-existing demo traces continue to deserialise without anyone having
to roll back to the legacy commit.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType
from typing import Optional, Sequence

_OLD_PREFIX = "rdagent.components.coder.factor_coder.CoSTEER"
_NEW_PREFIX = "rdagent.components.coder.CoSTEER"


class _CoSTEERPathRedirector(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``<OLD_PREFIX>[.submodule]`` imports against the new package."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname != _OLD_PREFIX and not fullname.startswith(_OLD_PREFIX + "."):
            return None
        new_name = _NEW_PREFIX + fullname[len(_OLD_PREFIX):]
        target_module = importlib.import_module(new_name)
        sys.modules[fullname] = target_module
        spec = importlib.util.spec_from_loader(fullname, self, is_package=hasattr(target_module, "__path__"))
        return spec

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Optional[ModuleType]:
        # The module has already been populated by find_spec via the redirect
        # target, so we just hand it back to the import machinery.
        return sys.modules.get(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        # No-op: the redirect target is fully initialised already.
        return None


def install_compat_redirector() -> None:
    """Install the redirector once; safe to call multiple times."""
    if not any(isinstance(f, _CoSTEERPathRedirector) for f in sys.meta_path):
        sys.meta_path.append(_CoSTEERPathRedirector())
