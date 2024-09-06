import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch


class TestRDAgentImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rdagent_directory = Path(__file__).resolve().parent.parent.parent
        cls.modules = list(cls.import_all_modules_from_directory(cls.rdagent_directory))

    @staticmethod
    def import_all_modules_from_directory(directory):
        for file in directory.joinpath("rdagent").rglob("*.py"):
            if "meta_tpl" in str(file):
                continue
            if "_template" in str(file):
                continue
            if "main" in str(file):
                continue
            yield str(file)[str(file).index("rdagent") : -3].replace("/", ".")

    @patch('sys.argv', ['__main__.py'])
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
