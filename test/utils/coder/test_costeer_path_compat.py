"""Regression test for the pre-refactor CoSTEER import-path shim (see #1331).

The ``demo_traces`` archive linked from the README was pickled before
``rdagent.components.coder.factor_coder.CoSTEER`` was relocated to
``rdagent.components.coder.CoSTEER``. Loading those pickles in
``rdagent ui`` would otherwise fail with ``ModuleNotFoundError`` for the
old module path. The shim installs a meta-path finder that redirects the
old path to the new package.
"""

import importlib
import pickle
import sys
import unittest


class TestCoSTEERPathCompat(unittest.TestCase):
    OLD = "rdagent.components.coder.factor_coder.CoSTEER"
    NEW = "rdagent.components.coder.CoSTEER"

    def setUp(self) -> None:
        # Importing the package triggers install_compat_redirector().
        import rdagent.components.coder.factor_coder  # noqa: F401

    def test_old_package_path_resolves_to_new_package(self) -> None:
        old = importlib.import_module(self.OLD)
        new = importlib.import_module(self.NEW)
        self.assertIs(old, new)

    def test_old_submodule_paths_resolve_to_new_submodules(self) -> None:
        for sub in ["evaluators", "evolving_strategy", "knowledge_management", "task", "config"]:
            with self.subTest(submodule=sub):
                old = importlib.import_module(f"{self.OLD}.{sub}")
                new = importlib.import_module(f"{self.NEW}.{sub}")
                self.assertIs(old, new)

    def test_pickle_referencing_old_module_path_round_trips(self) -> None:
        # Locate a real class from the new module path…
        new_evaluators = importlib.import_module(f"{self.NEW}.evaluators")
        # Pick any concrete class exposed at module scope.
        target_cls = next(
            (
                obj
                for name, obj in vars(new_evaluators).items()
                if isinstance(obj, type) and obj.__module__ == new_evaluators.__name__
            ),
            None,
        )
        self.assertIsNotNone(
            target_cls,
            "Expected at least one concrete class on the CoSTEER evaluators module",
        )

        # …re-stamp it as if it had been defined under the *old* import path
        # (this is what a pre-refactor pickle records in its serialised form).
        original_module = target_cls.__module__
        try:
            target_cls.__module__ = f"{self.OLD}.evaluators"

            # Drop any cached old-path module entries so we exercise the
            # full meta-path lookup, the same way pickle.load() does on a
            # fresh process.
            for key in [k for k in sys.modules if k == self.OLD or k.startswith(self.OLD + ".")]:
                del sys.modules[key]

            data = pickle.dumps(target_cls)
            loaded = pickle.loads(data)
            self.assertIs(loaded, target_cls)
        finally:
            target_cls.__module__ = original_module


if __name__ == "__main__":
    unittest.main()
