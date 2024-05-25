linting_system_prompt_template = "You are a software engineer. \
You can write code to a high standard and are adept at solving {language} linting problems."


user_get_makefile_lint_commands_template = """
You get a Makefile which contains some linting rules. Here are its content ```{file_text}```
Please find executable commands about linting from it.

Please response with following json template:
{{
    "commands": <python -m xxx --params >,
}}
"""

user_get_files_contain_lint_commands_template = """
You get a file list of a repository. \
Some file maybe contain linting rules or linting commands which defined by repo authors.
Here are the file list:
```
{file_list}
```

Please find all files maybe correspond to linting from it.
Please response with following json template:
{{
    "files": </path/to/file>,
}}
"""


generate_lint_command_template = """
Please generate a command to lint or format a {language} repository.
Here are some information about different linting tools ```{linting_tools}```
"""


suffix2language_template = """
Here are the files suffix in one code repo: {suffix}.
Please tell me the programming language used in this repo and which language has linting-tools.
Your response should follow this template:
{{
    "languages": <languages list>,
    "languages_with_linting_tools": <languages with lingting tools list>
}}
"""


session_start_template = """
Please modify the Python code based on the lint info.
Due to the length of the code, I will first tell you the entire code, and then each time I ask a question, \
I will extract a portion of the code and tell you the error information contained in this code segment.
You need to fix the corresponding error in the code segment \
and return the code that can replace the corresponding code segment.

The Python code is from a complete Python project file. Each line of the code is annotated with a line number, \
separated from the original code by three characters ("<white space>|<white space>"). The vertical bars are aligned.
Here is the complete code, please be prepared to fix it:
```Python
{code}
```
"""


session_normal_template = """Please modify this code snippet based on the lint info. Here is the code snippet:
```Python
{code}
```

-----Lint info-----
{lint_info}
-------------------

The lint info contains one or more errors. \
Different errors are separated by blank lines. Each error follows this format:
-----Lint info format-----
<Line Number>:<Error Start Position> <Error Code> <Error Message>
<Error Context (multiple lines)>
<Helpful Information (last line)>
--------------------------
The error code is an abbreviation set by the checker for ease of describing the error. \
The error context includes the relevant code around the error, and the helpful information suggests possible fixes.

Please simply reply the code after you fix all linting errors.
The code you return does not require line numbers, \
and should just replace the code I provided you, and does not require comments.
Please wrap your code with following format:

```python
<your code..>
```
"""


user_template_for_code_snippet = """Please modify the Python code based on the lint info.
-----Python Code-----
{code}
---------------------

-----Lint info-----
{lint_info}
-------------------

The Python code is a snippet from a complete Python project file. \
Each line of the code is annotated with a line number, \
separated from the original code by three characters ("<white space>|<white space>"). \
The vertical bars are aligned.

The lint info contains one or more errors. Different errors are separated by blank lines. \
Each error follows this format:
-----Lint info format-----
<Line Number>:<Error Start Position> <Error Code> <Error Message>
<Error Context (multiple lines)>
<Helpful Information (last line)>
--------------------------
The error code is an abbreviation set by the checker for ease of describing the error. \
The error context includes the relevant code around the error, and the helpful information suggests possible fixes.

Please simply reply the code after you fix all linting errors.
The code you return does not require line numbers, \
and should just replace the code I provided you, and does not require comments.
Please wrap your code with following format:

```python
<your code..>
```
"""
