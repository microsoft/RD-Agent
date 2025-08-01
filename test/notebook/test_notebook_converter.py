import json
import os
import unittest

from rdagent.components.coder.data_science.share.notebook import NotebookConverter

test_files_dir = os.path.join(os.path.dirname(__file__), "testfiles")


def normalize_nb_json_for_comparison(nb_json_str):
    nb_json = json.loads(nb_json_str)
    for cell in nb_json["cells"]:
        if "id" in cell:
            cell.pop("id", None)
    return json.dumps(nb_json, indent=4)


class TestNotebookConverter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.converter = NotebookConverter()
        self.maxDiff = None

    def test_validation_pass(self):
        with open(os.path.join(test_files_dir, "main.py"), "r") as f:
            code = f.read()
            result = self.converter.validate_code_format(code)
            self.assertIsNone(result, "Code format should be valid")

    def test_validation_missing_main_fn(self):
        with open(os.path.join(test_files_dir, "main_missing_main_fn.py"), "r") as f:
            code = f.read()
            result = self.converter.validate_code_format(code)
            self.assertEqual(
                result,
                "[Error] No main function found in the code. Please ensure that the main function is defined and contains the necessary print statements to divide sections.",
            )

    def test_validation_missing_sections(self):
        with open(os.path.join(test_files_dir, "main_missing_sections.py"), "r") as f:
            code = f.read()
            result = self.converter.validate_code_format(code)
            self.assertEqual(
                result,
                "[Error] No sections found in the code. Expected to see 'print(\"Section: <section name>\")' as section dividers. Also make sure that they are actually run and not just comments.",
            )

    def test_argparse_happy_path(self):
        code = """import argparse
parser = argparse.ArgumentParser(description='Test script')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

def main():
    print(args.debug)
    print("Section: Data Loading")
    # Load dataset from CSV into a DataFrame
    load_data()

if __name__ == "__main__":
    main()"""
        notebookJson = json.loads(
            self.converter.convert(
                task=None,
                code=code,
                stdout="",
                use_debug_flag=True,
            )
        )
        self.assertEqual(
            "".join(notebookJson["cells"][0]["source"]),
            """import sys
# hack to allow argparse to work in notebook
sys.argv = ["main.py", "--debug"]

import argparse
parser = argparse.ArgumentParser(description='Test script')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

print(args.debug)""",
        )

        self.assertEqual(
            "".join(notebookJson["cells"][1]["source"]),
            """## Data Loading
Load dataset from CSV into a DataFrame
""",
        )
        self.assertEqual(
            "".join(notebookJson["cells"][2]["source"]),
            """print("Section: Data Loading")
load_data()""",
        )

    def test_argparse_with_dupe_sys(self):
        code = """import argparse
import sys
parser = argparse.ArgumentParser(description='Test script')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

print(sys)

def main():
    print(args.debug)
    print("Section: Data Loading")
    # Load dataset from CSV into a DataFrame
    load_data()

if __name__ == "__main__":
    main()"""
        notebookJson = json.loads(
            self.converter.convert(
                task=None,
                code=code,
                stdout="",
                use_debug_flag=True,
            )
        )
        self.assertEqual(
            "".join(notebookJson["cells"][0]["source"]),
            """import sys
# hack to allow argparse to work in notebook
sys.argv = ["main.py", "--debug"]

import argparse
parser = argparse.ArgumentParser(description='Test script')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

print(sys)

print(args.debug)""",
        )

        self.assertEqual(
            "".join(notebookJson["cells"][1]["source"]),
            """## Data Loading
Load dataset from CSV into a DataFrame
""",
        )
        self.assertEqual(
            "".join(notebookJson["cells"][2]["source"]),
            """print("Section: Data Loading")
load_data()""",
        )

    def test_convert(self):
        with open(os.path.join(test_files_dir, "main.py"), "r") as f:
            code = f.read()
            notebookJson = self.converter.convert(
                task=None,
                code=code,
                stdout="",
                # outfile=os.path.join(test_files_dir, "main.ipynb"), # Uncomment this to save to the file
            )
        with open(os.path.join(test_files_dir, "main.ipynb"), "r") as f:
            expected_notebook = f.read()
            self.assertEqual(
                normalize_nb_json_for_comparison(notebookJson),
                normalize_nb_json_for_comparison(expected_notebook),
                "Converted notebook should match expected output",
            )

    def test_convert_2(self):
        with open(os.path.join(test_files_dir, "main2.py"), "r") as f:
            code = f.read()
            notebookJson = self.converter.convert(
                task=None,
                code=code,
                stdout="",
                # outfile=os.path.join(test_files_dir, "main2.ipynb"), # Uncomment this to save to the file
            )
        with open(os.path.join(test_files_dir, "main2.ipynb"), "r") as f:
            expected_notebook = f.read()
            self.assertEqual(
                normalize_nb_json_for_comparison(notebookJson),
                normalize_nb_json_for_comparison(expected_notebook),
                "Converted notebook should match expected output",
            )


if __name__ == "__main__":
    unittest.main()
    # pytest test/notebook/test_notebook_converter.py
