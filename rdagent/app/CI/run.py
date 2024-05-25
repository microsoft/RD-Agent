"""

"""
import json
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import IS_LINE_JUNK, ndiff
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast

from rdagent.core.evolving_framework import (
    Evaluator,
    EvoAgent,
    EvolvableSubjects,
    EvolvingStrategy,
    EvoStep,
    Feedback,
    Knowledge,
)
from rdagent.oai.llm_utils import APIBackend
from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
import datetime

from .prompts import (
    linting_system_prompt_template,
    session_normal_template,
    session_start_template,
)


@dataclass
class CIError:
    raw_str: str
    file_path: Union[Path, str]
    line: int
    column: int
    code: str
    msg: str
    hint: str

    def to_dict(self):
        return self.__dict__

@dataclass
class CIFeedback(Feedback):
    errors: Dict[str, List[CIError]]


@dataclass
class FixRecord:
    skipped_errors: List[CIError]
    directly_fixed_errors: List[CIError]
    manually_fixed_errors: List[CIError]
    manual_instructions: Dict[str, List[CIError]]

    def to_dict(self):
        return {
            "skipped_errors": [error.to_dict() for error in self.skipped_errors],
            "directly_fixed_errors": [error.to_dict() for error in self.directly_fixed_errors],
            "manually_fixed_errors": [error.to_dict() for error in self.manually_fixed_errors],
            "manual_instructions": {key: [error.to_dict() for error in errors] for key, errors in self.manual_instructions.items()},
        }


class CodeFile:
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)
        self.load()


    def load(self) -> None:
        code = self.path.read_text(encoding="utf-8")
        self.code_lines = code.split("\n")

        # add line number
        self.lineno = len(self.code_lines)
        self.lineno_width = len(str(self.lineno))
        self.code_lines_with_lineno = []
        for i, code_line in enumerate(self.code_lines):
            self.code_lines_with_lineno.append(f"{i+1: >{self.lineno_width}} | {code_line}")


    def get(self, start = 0, end = None, add_line_number: bool = False, return_list: bool = False) -> Union[List[str], str]:
        start -= 1
        if start < 0: start = 0
        end = self.lineno if end is None else end-1

        res = self.code_lines_with_lineno[start:end] if add_line_number else self.code_lines[start:end]

        return res if return_list else "\n".join(res)


    def apply_changes(self, changes: List[Tuple[int, int, str]]) -> None:
        offset = 0
        for start, end, code in changes:
            start -= 1
            if start < 0: start = 0
            end -= 1

            new_code = code.split("\n")
            self.code_lines[start+offset:end+offset] = new_code
            offset += len(new_code) - (end - start)

        self.path.write_text("\n".join(self.code_lines), encoding="utf-8")
        self.load()


    def __str__(self):
        return f"{self.path}"


class Repo(EvolvableSubjects):
    def __init__(self, project_path: Union[Path, str], **kwargs):
        self.params = kwargs
        self.project_path = Path(project_path)
        git_ignored_output = subprocess.check_output(
            "git status --ignored -s",
            shell=True,
            cwd=project_path,
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
        git_ignored_files = [
            (self.project_path / Path(line[3:])).resolve()
            for line in git_ignored_output.split("\n")
            if line.startswith("!!")
        ]

        files = [
            file
            for file in self.project_path.glob("**/*")
            if file.is_file()
            and not any(str(file).startswith(str(path)) for path in git_ignored_files)
            and ".git/" not in str(file)
            and file.suffix == ".py"
        ]
        self.files = {file: CodeFile(file) for file in files}

        self.fix_records: Dict[str, FixRecord] | None = None


@dataclass
class RuffRule:
    """
    {
        "name": "missing-trailing-comma",
        "code": "COM812",
        "linter": "flake8-commas",
        "summary": "Trailing comma missing",
        "message_formats": [
            "Trailing comma missing"
        ],
        "fix": "Fix is always available.",
        "explanation": "## What it does\nChecks for the absence of trailing commas.\n\n## Why is this bad?\nThe presence of a trailing comma can reduce diff size when parameters or\nelements are added or removed from function calls, function definitions,\nliterals, etc.\n\n## Example\n```python\nfoo = {\n    \"bar\": 1,\n    \"baz\": 2\n}\n```\n\nUse instead:\n```python\nfoo = {\n    \"bar\": 1,\n    \"baz\": 2,\n}\n```\n",
        "preview": false
    }
    """
    name: str
    code: str
    linter: str
    summary: str
    message_formats: List[str]
    fix: str
    explanation: str
    preview: bool


class RuffEvaluator(Evaluator):
    """The error message are generated by
    `python -m ruff .  --exclude FinCo,finco,fincov1 --ignore ANN101,TCH003,D,ERA001`
    """

    def __init__(self, command: str = None):
        if command is None:
            self.command = "ruff check . --no-fix --output-format full"
        else:
            self.command = command


    def explain_rule(self, error_code: str) -> RuffRule:
        explain_command = "ruff rule {error_code} --output-format json"
        try:
            out = subprocess.check_output(
                explain_command.format(error_code=error_code),
                shell=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        return json.loads(out.decode())


    def evaluate(self, evo: Repo, **kwargs) -> CIFeedback:
        """Simply run ruff to get the feedbacks."""
        try:
            out = subprocess.check_output(
                self.command,
                shell=True,
                cwd=evo.project_path,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        """ruff output format:
        src/finco/cli.py:9:5: ANN201 Missing return type annotation for public function `main`
        |
        9 | def main(prompt=None):
        |     ^^^^ ANN201
        10 |     load_dotenv(verbose=True, override=True)
        11 |     wm = WorkflowManager()
        |
        = help: Add return type annotation: `None`
        """

        # extract error info
        pattern = r"(([^\n]*):(\d+):(\d+): (\w+) ([^\n]*)\n(.*?))\n\n"
        matches = re.findall(pattern, out.decode(), re.DOTALL)

        errors = defaultdict(list)
        for match in matches:
            raw_str, file_path, line_number, column_number, error_code, error_message, error_hint = match
            error = CIError(raw_str=raw_str,
                            file_path=file_path,
                            line=int(line_number),
                            column=int(column_number),
                            code=error_code,
                            msg=error_message,
                            hint=error_hint)
            errors[file_path].append(error)

        return CIFeedback(errors=errors)

class MypyEvaluator(Evaluator):

    def __init__(self, command: str = None):
        if command is None:
            self.command = "mypy . --explicit-package-bases"
        else:
            self.command = command

    def evaluate(self, evo: Repo, **kwargs) -> CIFeedback:
        try:
            out = subprocess.check_output(
                self.command,
                shell=True,
                cwd=evo.project_path,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        return CIFeedback(cast(str, out).decode("utf-8"))


class CIEvoStr(EvolvingStrategy):

    def evolve(
        self,
        evo: Repo,
        evolving_trace: List[EvoStep] = [],
        knowledge_l: List[Knowledge] = [],
        **kwargs,
    ) -> Repo:
        api = APIBackend()
        system_prompt = linting_system_prompt_template.format(language="Python")

        if len(evolving_trace) > 0:
            last_feedback: CIFeedback = evolving_trace[-1].feedback
            fix_records: Dict[str, FixRecord] = defaultdict(lambda: FixRecord([], [], [], defaultdict(list)))
            # iterate by file
            for file_path, errors in last_feedback.errors.items():
                print(Rule(f"[cyan]Fixing {file_path}[/cyan]", style="bold cyan", align="left", characters="."))

                file = evo.files[evo.project_path / Path(file_path)]

                # Group errors based on position
                # TODO @bowen: Crossover between different groups after adding 3 lines of context
                groups: List[List[CIError]] = []
                near_errors = [errors[0]]
                for error in errors[1:]:
                    if error.line - near_errors[-1].line <= 6:
                        near_errors.append(error)
                    else:
                        groups.append(near_errors)
                        near_errors = [error]
                groups.append(near_errors)

                changes = []

                # generate changes
                for group_id, group in enumerate(groups, start=1):
                    session = api.build_chat_session(session_system_prompt=system_prompt)
                    session.build_chat_completion(session_start_template.format(code=file.get(add_line_number=True)))

                    print(f"[yellow]Fixing part {group_id}...[/yellow]\n")

                    start_line = group[0].line - 3
                    end_line = group[-1].line + 3 + 1
                    code_snippet_with_lineno = file.get(start_line, end_line, add_line_number=True, return_list=False)
                    code_snippet_lines = file.get(start_line, end_line, add_line_number=False, return_list=True)

                    # front_anchor_code = file.get(start_line-3, start_line, add_line_number=False, return_list=False)
                    # rear_anchor_code = file.get(end_line+1, end_line+3+1, add_line_number=False, return_list=False)

                    errors_str = "\n".join([f"{error.raw_str}\n" for error in group])

                    print(Panel.fit(Syntax("\n".join([f"{error.line}: {error.msg}" for error in group]), lexer="python", background_color="default"), title=f"{len(group)} Errors"))
                    # print(f"[bold yellow]original code:[/bold yellow]\n\n{code_snippet_with_lineno}")
                    print(Panel.fit(Syntax(code_snippet_with_lineno, lexer="python", background_color="default"), title="Original Code"))
                    user_prompt = session_normal_template.format(
                        code=code_snippet_with_lineno,
                        lint_info=errors_str,
                    )
                    res = session.build_chat_completion(user_prompt)

                    manual_fix_flag = False

                    while True:
                        try:
                            new_code = re.search(r".*```[Pp]ython\n(.*)\n```.*", res, re.DOTALL).group(1)
                            
                            # print repair status (code diff)
                            diff = ndiff(code_snippet_lines, new_code.split("\n"), linejunk=IS_LINE_JUNK)
                            table = Table(show_header=False, box=None)
                            table.add_column()
                            for i in diff:
                                if i.startswith("+"): table.add_row(Text(i, style="green"))
                                elif i.startswith("-"): table.add_row(Text(i, style="red"))
                                elif i.startswith("?"): table.add_row(Text(i, style="yellow"))
                                else: table.add_row(Syntax(i, lexer="python", background_color="default"))
                            print(Panel.fit(table, title="Repair Status"))
                        except Exception as e:
                            print(f"[red]Error[/red]: {e}")

                        operation = input("Input your operation: ")
                        if operation == "s" or operation == "skip":
                            fix_records[file_path].skipped_errors.extend(group)
                            break
                        if operation == "a" or operation == "apply":
                            if manual_fix_flag:
                                fix_records[file_path].manually_fixed_errors.extend(group)
                            else:
                                fix_records[file_path].directly_fixed_errors.extend(group)

                            changes.append((start_line, end_line, new_code))
                            break
                        
                        manual_fix_flag = True
                        fix_records[file_path].manual_instructions[operation].extend(group)
                        res = session.build_chat_completion(('There are some problems with the code you provided, '
                                            'please follow the instruction below to fix it again and return.\n'
                                            f'Instruction: {operation}'))

                # apply changes
                file.apply_changes(changes)

            evo.fix_records = fix_records

        return evo


DIR = None
while True:
    DIR = Prompt.ask("Please input the project directory")
    DIR = Path(DIR)
    if DIR.exists():
        break
    else:
        print("Invalid directory. Please try again.")

start_time = time.time()
start_timestamp = datetime.datetime.now().strftime("%m%d%H%M")

evo = Repo(DIR)
eval = RuffEvaluator()
estr = CIEvoStr()
rag = None  # RAG is not enable firstly.
ea = EvoAgent(estr, rag=rag)
ea.step_evolving(evo, eval)
while True:
    print(Rule(f"Round {len(ea.evolving_trace)} repair", style="blue"))
    evo: Repo = ea.step_evolving(evo, eval)

    fix_records = evo.fix_records
    filename = f"{DIR.name}_{start_timestamp}_fix_records_{len(ea.evolving_trace)}.json"
    json.dump(fix_records, open(filename, "w"), indent=4)

    # Count the number of skipped errors
    skipped_errors_count = 0
    directly_fixed_errors_count = 0
    manually_fixed_errors_count = 0
    skipped_errors_code_count = defaultdict(int)
    directly_fixed_errors_code_count = defaultdict(int)
    manually_fixed_errors_code_count = defaultdict(int)
    for record in fix_records.values():
        skipped_errors_count += len(record.skipped_errors)
        directly_fixed_errors_count += len(record.directly_fixed_errors)
        manually_fixed_errors_count += len(record.manually_fixed_errors)
        for error in record.skipped_errors:
            skipped_errors_code_count[error.code] += 1
        for error in record.directly_fixed_errors:
            directly_fixed_errors_code_count[error.code] += 1
        for error in record.manually_fixed_errors:
            manually_fixed_errors_code_count[error.code] += 1

    skipped_errors_statistics = ""
    directly_fixed_errors_statistics = ""
    manually_fixed_errors_statistics = ""
    for code, count in sorted(skipped_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        skipped_errors_statistics += f"{count: >5} {code: >10} {eval.explain_rule(code).summary}\n"
    for code, count in sorted(directly_fixed_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        directly_fixed_errors_statistics += f"{count: >5} {code: >10} {eval.explain_rule(code).summary}\n"
    for code, count in sorted(manually_fixed_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        manually_fixed_errors_statistics += f"{count: >5} {code: >10} {eval.explain_rule(code).summary}\n"

    # Create a table to display the counts and ratios
    table = Table(title="Error Fix Statistics")
    table.add_column("Type")
    table.add_column("Statistics")
    table.add_column("Count")
    table.add_column("Ratio")

    total_errors_count = skipped_errors_count + directly_fixed_errors_count + manually_fixed_errors_count
    table.add_row("Total Errors", "", str(total_errors_count), "")
    table.add_row("Skipped Errors", skipped_errors_statistics, str(skipped_errors_count), f"{skipped_errors_count / total_errors_count:.2%}")
    table.add_row("Directly Fixed Errors", directly_fixed_errors_statistics, str(directly_fixed_errors_count), f"{directly_fixed_errors_count / total_errors_count:.2%}")
    table.add_row("Manually Fixed Errors", manually_fixed_errors_statistics, str(manually_fixed_errors_count), f"{manually_fixed_errors_count / total_errors_count:.2%}")

    print(table)
    operation = Prompt.ask("Start next round? (y/n)", choices=["y", "n"])
    if operation == "n": break


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

""" Please commit it by hand... and then run the next round
git add -u
git commit --no-verify  -v
"""
