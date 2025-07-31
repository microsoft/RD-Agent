import unittest

from rdagent.components.coder.data_science.share.util import (
    extract_function_body,
    split_code_sections,
    split_output_sections,
    extract_comment_under_first_print,
    extract_first_section_name_from_code,
    extract_first_section_name_from_output,
    is_function_called,
    remove_function,
    remove_main_block,
    split_code_and_output_into_sections
)

class TestExtractFunctionBody(unittest.TestCase):
    def test_happy_path(self):
        code = S([
            "def main():",
            "    print('Section: Data Loading')",
            "    # Load data",
            "    data = load_data()",
            "",
        ])
        extracted = extract_function_body(code, "main")
        expected = S([
            "print('Section: Data Loading')",
            "# Load data",
            "data = load_data()",
        ])
        self.assertEqual(extracted, expected)

    def test_happy_path_complex(self):
        code = S([
            "import pandas as pd",
            "",
            "print('main()')",
            "",
            "def foo():",
            "    print('Section: Foo')",
            "",
            "def mainfunc():",
            "    print('Section: Data Loading 2')",
            "    # Load data 2",
            "    data2 = load_data()",
            "",
            "def main():",
            "    print('Section: Data Loading')",
            "    # Load data",
            "    data = load_data()",
            "",
            "def bar():",
            "    print('Section: Foo')",
            "",
            "main()"
        ])
        extracted = extract_function_body(code, "main")
        expected = S([
            "print('Section: Data Loading')",
            "# Load data",
            "data = load_data()",
        ])
        self.assertEqual(extracted, expected)

    def test_empty(self):
        extracted = extract_function_body("", "main")
        expected = None
        self.assertEqual(extracted, expected)

    def test_missing_func(self):
        code = S([
            "def foo():",
            "    print('Section: Data Loading')",
            "    # Load data",
            "    data = load_data()",
            "",
        ])
        extracted = extract_function_body(code, "main")
        expected = None
        self.assertEqual(extracted, expected)

class TestSplitCodeSections(unittest.TestCase):
    def test_happy_path(self):
        code = S([
            "# This is the main function",
            "setup_workspace()",
            "print('Section: Data Loading')",
            "# Load data",
            "data = load_data()",
            'print("Section: Data Processing")',
            "# Process data",
            "processed_data = process_data(data)",
        ])
        header, sections = split_code_sections(code)
        self.assertEqual(header, S([
            "# This is the main function",
            "setup_workspace()",
        ]))
        self.assertListEqual(sections, [
            S([
                "print('Section: Data Loading')",
                "# Load data",
                "data = load_data()",
            ]),
            S([
                'print("Section: Data Processing")',
                "# Process data",
                "processed_data = process_data(data)",
            ])
        ])

    def test_happy_path_no_header(self):
        code = S([
            "print('Section: Setup')",
            "# This is the main function",
            "setup_workspace()",
            "print('Section: Data Loading')",
            "# Load data",
            "data = load_data()",
            "print('Section: Data Processing')",
            "# Process data",
            "processed_data = process_data(data)",
        ])
        header, sections = split_code_sections(code)
        self.assertEqual(header, None)
        self.assertListEqual(sections, [
            S([
                "print('Section: Setup')",
                "# This is the main function",
                "setup_workspace()",
            ]),
            S([
                "print('Section: Data Loading')",
                "# Load data",
                "data = load_data()",
            ]),
            S([
                "print('Section: Data Processing')",
                "# Process data",
                "processed_data = process_data(data)",
            ])
        ])

    def test_wrong_format(self):
        code = S([
            "# This is the main function",
            "setup_workspace()",
            "print('A Section: Data Loading')",
            "# Load data",
            "data = load_data()",
            's = """print(\'Section: Data Processing\')"""',
            "# Process data",
            "processed_data = process_data(data)",
        ])
        header, sections = split_code_sections(code)
        self.assertEqual(header, S([
            "# This is the main function",
            "setup_workspace()",
            "print('A Section: Data Loading')",
            "# Load data",
            "data = load_data()",
            's = """print(\'Section: Data Processing\')"""',
            "# Process data",
            "processed_data = process_data(data)",
        ]))
        self.assertListEqual(sections, [])

    def test_empty(self):
        code = ""
        header, sections = split_code_sections(code)
        self.assertEqual(header, None)
        self.assertListEqual(sections, [])

    def test_single_no_sections(self):
        code = "print('foo')"
        header, sections = split_code_sections(code)
        self.assertEqual(header, "print('foo')")
        self.assertListEqual(sections, [])

    def test_single_with_section(self):
        code = "print('Section: foo')"
        header, sections = split_code_sections(code)
        self.assertEqual(header, None)
        self.assertListEqual(sections, ["print('Section: foo')"])

    def test_no_sections(self):
        code = S([
            "# This is the main function",
            "setup_workspace()",
            "# Load data",
            "data = load_data()",
            "# Process data",
            "processed_data = process_data(data)",
        ])
        header, sections = split_code_sections(code)
        self.assertEqual(header, S([
            "# This is the main function",
            "setup_workspace()",
            "# Load data",
            "data = load_data()",
            "# Process data",
            "processed_data = process_data(data)",
        ]))
        self.assertListEqual(sections, [])

class TestSplitOutputSections(unittest.TestCase):
    def test_happy_path(self):
        output = S([
            "Setting up workspace...",
            "Section: Data Loading",
            "Loading data...",
            "Section: Data Processing",
            "Processing data...",
        ])
        header, sections = split_output_sections(output)
        self.assertEqual(header, S([
            "Setting up workspace...",
        ]))
        self.assertListEqual(sections, [
            S([
                "Section: Data Loading",
                "Loading data..."
            ]),
            S([
                "Section: Data Processing",
                "Processing data...",
            ])
        ])

    def test_happy_path_no_header(self):
        output = S([
            "Section: Setup",
            "Setting up workspace...",
            "Section: Data Loading",
            "Loading data...",
            "Section: Data Processing",
            "Processing data...",
        ])
        header, sections = split_output_sections(output)
        self.assertEqual(header, None)
        self.assertListEqual(sections, [
            S([
                "Section: Setup",
                "Setting up workspace...",
            ]),
            S([
                "Section: Data Loading",
                "Loading data..."
            ]),
            S([
                "Section: Data Processing",
                "Processing data...",
            ])
        ])

    def test_wrong_format(self):
        output = S([
            "Setting up workspace...",
            "Wrong Section: Data Loading",
            "Loading data...",
            "Wrong Section: Data Processing",
            "Processing data...",
        ])
        header, sections = split_output_sections(output)
        self.assertEqual(header, S([
            "Setting up workspace...",
            "Wrong Section: Data Loading",
            "Loading data...",
            "Wrong Section: Data Processing",
            "Processing data...",
        ]))
        self.assertListEqual(sections, [])

    def test_empty(self):
        output = ""
        header, sections = split_output_sections(output)
        self.assertEqual(header, None)
        self.assertListEqual(sections, [])

    def test_single_no_sections(self):
        output = "foo"
        header, sections = split_output_sections(output)
        self.assertEqual(header, "foo")
        self.assertListEqual(sections, [])

    def test_single_with_section(self):
        output = "Section: foo"
        header, sections = split_output_sections(output)
        self.assertEqual(header, None)
        self.assertListEqual(sections, ["Section: foo"])

    def test_no_sections(self):
        output = S([
            "Setting up workspace...",
            "Loading data..."
            "Processing data...",
        ])
        header, sections = split_output_sections(output)
        self.assertEqual(header, S([
            "Setting up workspace...",
            "Loading data..."
            "Processing data...",
        ]))
        self.assertListEqual(sections, [])

class TestExtractSectionComments(unittest.TestCase):
    def test_happy_path(self):
        code = S([
            "print('Section: Data Loading')",
            "# Load data",
            "data = load_data()",
            "print('Section: Data Processing')",
            "# Process data",
            "processed_data = process_data(data)",
        ])
        comments, cleaned = extract_comment_under_first_print(code)
        self.assertEqual(comments, "Load data")
        self.assertEqual(cleaned, S([
            "print('Section: Data Loading')",
            "data = load_data()",
            "print('Section: Data Processing')",
            "# Process data",
            "processed_data = process_data(data)",
        ]))

    def test_happy_path_multiline(self):
        code = S([
            "print('Section: Data Loading')",
            "# Load data",
            "# This section loads some data",
            "data = load_data()",
            "print('Section: Data Processing')",
            "# Process data",
            "processed_data = process_data(data)",
        ])
        comments, cleaned = extract_comment_under_first_print(code)
        self.assertEqual(comments, S(["Load data", "This section loads some data"]))
        self.assertEqual(cleaned, S([
            "print('Section: Data Loading')",
            "data = load_data()",
            "print('Section: Data Processing')",
            "# Process data",
            "processed_data = process_data(data)",
        ]))

    def test_no_comment(self):
        code = S([
            "print('Section: Data Loading')",
            "data = load_data()",
            "print('Section: Data Processing')",
            "# Process data",
            "processed_data = process_data(data)",
        ])
        comments, cleaned = extract_comment_under_first_print(code)
        self.assertEqual(comments, None)
        self.assertEqual(cleaned, S([
            "print('Section: Data Loading')",
            "data = load_data()",
            "print('Section: Data Processing')",
            "# Process data",
            "processed_data = process_data(data)",
        ]))

    def test_arbitrary_print_happy_path(self):
        code = S([
            "print('No section here')",
            "# Just a comment",
            "data = load_data()",
        ])
        comments, cleaned = extract_comment_under_first_print(code)
        self.assertEqual(comments, "Just a comment")
        self.assertEqual(cleaned, S([
            "print('No section here')",
            "data = load_data()",
        ]))

    def test_empty_string(self):
        code = ""
        comments, cleaned = extract_comment_under_first_print(code)
        self.assertEqual(comments, None)
        self.assertEqual(cleaned, "")

class TestExtractFirstSectionNameFromCode(unittest.TestCase):
    def test_happy_path(self):
        code = S([
            "print('Section: Data Loading')",
            "# Load data",
            "data = load_data()",
            "print('Section: Data Processing')",
            "# Process data",
            "processed_data = process_data(data)",
        ])
        section_name = extract_first_section_name_from_code(code)
        self.assertEqual(section_name, "Data Loading")

    def test_no_section(self):
        code = S([
            "print('No section here')",
            "# Just a comment",
            "data = load_data()",
        ])
        section_name = extract_first_section_name_from_code(code)
        self.assertEqual(section_name, None)

    def test_empty_string(self):
        code = ""
        section_name = extract_first_section_name_from_code(code)
        self.assertEqual(section_name, None)

class TestExtractFirstSectionNameFromOutput(unittest.TestCase):
    def test_happy_path(self):
        output = S([
            "Setting up workspace...",
            "Section: Data Loading",
            "Loading data...",
            "Section: Data Processing",
            "Processing data...",
        ])
        section_name = extract_first_section_name_from_output(output)
        self.assertEqual(section_name, "Data Loading")

    def test_no_section(self):
        output = S([
            "Setting up workspace...",
            "Loading data...",
            "Processing data...",
        ])
        section_name = extract_first_section_name_from_output(output)
        self.assertEqual(section_name, None)

    def test_empty_string(self):
        output = ""
        section_name = extract_first_section_name_from_output(output)
        self.assertEqual(section_name, None)

class TestIsFunctionCalled(unittest.TestCase):
    def test_happy_path(self):
        code = S([
            "def main():",
            "    print('Hello World')",
            "",
            "main()"
        ])
        self.assertTrue(is_function_called(code, "main"))

    def test_happy_path_with_args(self):
        code = S([
            "main(123, 'abc')",
        ])
        self.assertTrue(is_function_called(code, "main"))

    def test_happy_path_with_args_multiline(self):
        code = S([
            "main(",
            "   123,",
            "   'abc'",
            ")",
        ])
        self.assertTrue(is_function_called(code, "main"))

    def test_not_called(self):
        code = S([
            "def main():",
            "    print('Hello World')",
            "",
        ])
        self.assertFalse(is_function_called(code, "main"))

    def test_wrong_format(self):
        code = S([
            "def main():",
            "    print('Hello World')",
            "",
            "main2()"
        ])
        self.assertFalse(is_function_called(code, "main"))

    def test_empty_string(self):
        code = ""
        self.assertFalse(is_function_called(code, "main"))

class TestRemoveFunction(unittest.TestCase):
    def test_happy_path(self):
        code = S([
            "def main():",
            "    print('Hello World')",
            "",
            "main()"
        ])
        cleaned_code = remove_function(code, "main")
        expected_code = S([
            "",
            "main()"
        ])
        self.assertEqual(cleaned_code, expected_code)

    def test_function_does_not_exist(self):
        code = S([
            "def main2():",
            "    print('Hello World')",
            "",
            "main()"
        ])
        cleaned_code = remove_function(code, "main")
        expected_code = S([
            "def main2():",
            "    print('Hello World')",
            "",
            "main()"
        ])
        self.assertEqual(cleaned_code, expected_code)

    def test_empty(self):
        code = ""
        cleaned_code = remove_function(code, "main")
        expected_code = ""
        self.assertEqual(cleaned_code, expected_code)

    def test_preserves_comments(self):
        code = S([
            "def main():",
            '    """'
            "    This is the main function.",
            '    """',
            "    print('Hello World')",
            "",
            "def main2():",
            '    """'
            "    This is the second main function.",
            '    """',
            "    print('Hello World')",
            "",
            "# Some comment",
            "main()"
        ])
        cleaned_code = remove_function(code, "main")
        expected_code = S([
            "",
            "def main2():",
            '    """'
            "    This is the second main function.",
            '    """',
            "    print('Hello World')",
            "",
            "# Some comment",
            "main()"
        ])
        self.assertEqual(cleaned_code, expected_code)

class TestRemoveMainBlock(unittest.TestCase):
    def test_happy_path(self):
        code = S([
            "if __name__ == '__main__':",
            "    main()",
        ])
        cleaned_code = remove_main_block(code)
        expected_code = ""
        self.assertEqual(cleaned_code, expected_code)

    def test_one_liner(self):
        code = S([
            "if __name__ == '__main__': main()",
        ])
        cleaned_code = remove_main_block(code)
        expected_code = ""
        self.assertEqual(cleaned_code, expected_code)

    def test_happy_path_arbitrary_content(self):
        code = S([
            "if __name__ == '__main__':",
            "    # foo",
            "    print('Hello World')",
            "    main()",
        ])
        cleaned_code = remove_main_block(code)
        expected_code = ""
        self.assertEqual(cleaned_code, expected_code)

    def test_block_does_not_exist(self):
        code = S([
            "if __name__ == '__foo__':",
            "    main()",
        ])
        cleaned_code = remove_main_block(code)
        expected_code = S([
            "if __name__ == '__foo__':",
            "    main()",
        ])
        self.assertEqual(cleaned_code, expected_code)

    def test_empty(self):
        code = ""
        cleaned_code = remove_main_block(code)
        expected_code = ""
        self.assertEqual(cleaned_code, expected_code)

class TestSplitCodeAndOutputIntoSections(unittest.TestCase):
    def test_happy_path(self):
        code = S([
            "# Some notebook comments",
            "import pandas as pd",
            "",
            "RANDOM_SEED = 42",
            ""
            "def setup():",
            "    print('Setting up workspace...')",
            "",
            "def load_data():",
            "    return []",
            "",
            "def process_data(data):",
            "    return data",
            "",
            "def main():",
            "    setup()",
            "    print('Section: Data Loading')",
            "    # Load data",
            "    data = load_data()",
            "",
            "    print('Section: Data Processing')",
            "    # Process data",
            "    processed_data = process_data(data)",
        ])
        output = S([
            "Setting up workspace...",
            "Section: Data Loading",
            "Loading data...",
            "Section: Data Processing",
            "Processing data...",
        ])
        sections = split_code_and_output_into_sections(code=code, stdout=output)
        self.assertEqual(len(sections), 3)
        self.assertDictEqual(sections[0], {
            "name": None,
            "comments": None,
            "code": S([
                "# Some notebook comments",
                "import pandas as pd",
                "",
                "RANDOM_SEED = 42",
                ""
                "def setup():",
                "    print('Setting up workspace...')",
                "",
                "setup()"
            ]),
            "output": S([
                "Setting up workspace..."
            ])
        })
        self.assertDictEqual(sections[1], {
            "name": "Data Loading",
            "comments": "Load data",
            "code": S([
                "def load_data():",
                "    return []",
                "",
                "print('Section: Data Loading')",
                "data = load_data()",
            ]),
            "output": S([
                "Section: Data Loading",
                "Loading data...",
            ])
        })
        self.assertDictEqual(sections[2], {
            "name": "Data Processing",
            "comments": "Process data",
            "code": S([
                "def process_data(data):",
                "    return data",
                "",
                "print('Section: Data Processing')",
                "processed_data = process_data(data)",
            ]),
            "output": S([
                "Section: Data Processing",
                "Processing data...",
            ])
        })

    def test_empty_code(self):
        code = ""
        output = S([
            "Setting up workspace...",
            "Section: Data Loading",
            "Loading data...",
            "Section: Data Processing",
            "Processing data...",
        ])
        sections = split_code_and_output_into_sections(code=code, stdout=output)
        self.assertEqual(len(sections), 3)
        self.assertDictEqual(sections[0], {
            "name": None,
            "comments": None,
            "code": "",
            "output": S([
                "Setting up workspace...",
            ])
        })
        self.assertDictEqual(sections[1], {
            "name": "Data Loading",
            "comments": None,
            "code": None,
            "output": S([
                "Section: Data Loading",
                "Loading data...",
            ])
        })
        self.assertDictEqual(sections[2], {
            "name": "Data Processing",
            "comments": None,
            "code": None,
            "output": S([
                "Section: Data Processing",
                "Processing data...",
            ])
        })

    def test_empty_outputs(self):
        code = S([
            "# Some notebook comments",
            "import pandas as pd",
            "",
            "RANDOM_SEED = 42",
            ""
            "def setup():",
            "    print('Setting up workspace...')",
            "",
            "def load_data():",
            "    return []",
            "",
            "def process_data(data):",
            "    return data",
            "",
            "def main():",
            "    setup()",
            "    print('Section: Data Loading')",
            "    # Load data",
            "    data = load_data()",
            "",
            "    print('Section: Data Processing')",
            "    # Process data",
            "    processed_data = process_data(data)",
        ])
        output = ""
        sections = split_code_and_output_into_sections(code=code, stdout=output)
        self.assertEqual(len(sections), 3)
        self.assertDictEqual(sections[0], {
            "name": None,
            "comments": None,
            "code": S([
                "# Some notebook comments",
                "import pandas as pd",
                "",
                "RANDOM_SEED = 42",
                ""
                "def setup():",
                "    print('Setting up workspace...')",
                "",
                "setup()"
            ]),
            "output": None
        })
        self.assertDictEqual(sections[1], {
            "name": "Data Loading",
            "comments": "Load data",
            "code": S([
                "def load_data():",
                "    return []",
                "",
                "print('Section: Data Loading')",
                "data = load_data()",
            ]),
            "output": None
        })
        self.assertDictEqual(sections[2], {
            "name": "Data Processing",
            "comments": "Process data",
            "code": S([
                "def process_data(data):",
                "    return data",
                "",
                "print('Section: Data Processing')",
                "processed_data = process_data(data)",
            ]),
            "output": None
        })

def S(s_arr):
    return "\n".join(s_arr)

if __name__ == "__main__":
    unittest.main()
    # pytest test/notebook/test_util.py
