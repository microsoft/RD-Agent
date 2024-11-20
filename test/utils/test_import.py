import importlib
import os
import unittest
from pathlib import Path

import pytest


@pytest.mark.offline
class TestRDAgentImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rdagent_directory = Path(__file__).resolve().parent.parent.parent
        cls.modules = list(cls.import_all_modules_from_directory(cls.rdagent_directory))

    @staticmethod
    def import_all_modules_from_directory(directory):
        for file in directory.joinpath("rdagent").rglob("*.py"):
            fstr = str(file)
            if "meta_tpl" in fstr:
                continue
            if "template" in fstr or "tpl" in fstr:
                continue
            if "model_coder" in fstr:
                continue
            if (
                fstr.endswith("rdagent/log/ui/app.py")
                or fstr.endswith("rdagent/app/cli.py")
                or fstr.endswith("rdagent/app/CI/run.py")
            ):
                # the entrance points
                continue

            yield fstr[fstr.index("rdagent") : -3].replace("/", ".")

    def test_import_modules(self):
        print(self.modules)
        for module_name in self.modules:
            with self.subTest(module=module_name):
                try:
                    print(module_name)
                    importlib.import_module(module_name)
                except Exception as e:
                    self.fail(f"Failed to import {module_name}: {e}")


if __name__ == "__main__":
    unittest.main()
