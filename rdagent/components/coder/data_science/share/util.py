import ast
import io
import re
import tokenize
from itertools import zip_longest
from typing import List, Optional, Set, Tuple, TypedDict


class CodeSection(TypedDict):
    """
    Represents a section of the original Python source code, to be converted to a notebook cell.
    """

    name: Optional[str]
    code: Optional[str]
    comments: Optional[str]
    output: Optional[str]


def extract_function_body(source_code: str, function_name: str) -> Optional[str]:
    """
    Extracts the body of a function from the source code.
    Returns None if the function is not found.

    Assumption: The function is multiline and defined at the top level.
    """
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            lines = source_code.splitlines()
            start = node.body[0].lineno
            end = node.body[-1].end_lineno
            body_lines = lines[start - 1 : end]
            indent_level = len(body_lines[0]) - len(body_lines[0].lstrip())
            return "\n".join(line[indent_level:] for line in body_lines)
    return None


def split_sections(
    text: str, section_header_regex: str, known_sections: Optional[list[str]] = None
) -> tuple[Optional[str], list[str], list[str]]:
    """
    Split text into sections based on the section headers.
    """
    sections = []
    section_names = []
    current_section = []
    next_section_name_index = 0
    for line in text.splitlines():
        match = re.match(section_header_regex, line)
        extracted_section_name = match.group(1).strip() if match else None
        if extracted_section_name and (
            not known_sections
            or (
                next_section_name_index < len(known_sections)
                and extracted_section_name == known_sections[next_section_name_index]
            )
        ):
            if current_section:
                sections.append("\n".join(current_section))
                current_section = []
            current_section.append(line)
            section_names.append(extracted_section_name)
            next_section_name_index += 1
        else:
            current_section.append(line)
    if current_section:
        sections.append("\n".join(current_section))

    # If the first section does not match the header regex, treat it as a header section.
    header_section = None
    if sections and not re.search(section_header_regex, sections[0]):
        header_section = sections[0]
        sections = sections[1:]

    return header_section, sections, section_names


def split_code_sections(source_code: str) -> tuple[Optional[str], list[str]]:
    """
    Split code into sections based on the section headers.
    """
    return split_sections(source_code, r'^print\(["\']Section: (.+)["\']\)')


def split_output_sections(stdout: str, known_sections: list[str]) -> tuple[Optional[str], list[str]]:
    """
    Split output into sections based on the section headers.
    """
    header_section, sections, _ = split_sections(stdout, r"^Section: (.+)", known_sections=known_sections)
    return header_section, sections


def extract_comment_under_first_print(source_code) -> tuple[Optional[str], str]:
    """
    Extract comments from the source code after the first print statement.
    """
    lines = source_code.splitlines()
    lines_to_remove = set()
    all_comments = []

    parsed = ast.parse(source_code)
    # Find the first print statement only
    first_print_lineno = None
    for node in ast.walk(parsed):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if getattr(node.value.func, "id", None) == "print":
                first_print_lineno = node.lineno
                break

    if first_print_lineno is None:
        # No print statement found, return empty comments and original code
        return None, source_code

    for i in range(first_print_lineno, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            comment_text = stripped.lstrip("# ").strip()
            all_comments.append(comment_text)
            lines_to_remove.add(i)
        elif stripped == "":
            continue
        elif i > first_print_lineno:
            break  # stop after hitting actual code line

    cleaned_lines = [line for idx, line in enumerate(lines) if idx not in lines_to_remove]
    cleaned_code = "\n".join(cleaned_lines)
    comments_str = "\n".join(all_comments) if all_comments else None

    return comments_str, cleaned_code


def extract_first_section_name_from_code(source_code):
    """
    Extract the first section name from the source code.
    """
    parsed = ast.parse(source_code)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if getattr(call.func, "id", None) == "print" and call.args:
                arg0 = call.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    # Match "Section: ..." pattern
                    m = re.match(r"Section:\s*(.+)", arg0.value)
                    if m:
                        return m.group(1).strip()
    return None


def extract_first_section_name_from_output(stdout: str) -> Optional[str]:
    """
    Extract the first section name from the output string.
    """
    match = re.search(r"Section:\s*(.+)", stdout)
    if match:
        return match.group(1).strip()
    return None


def is_function_called(source_code: str, func_name: str) -> bool:
    """
    Returns True if the function named `func_name` is called in `source_code`.
    """
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # For simple function calls like func()
            if isinstance(node.func, ast.Name) and node.func.id == func_name:
                return True

            # For calls like module.func()
            elif isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                return True
    return False


def remove_function(source_code: str, function_name: str) -> str:
    """
    Remove a function definition from the source code.
    """
    tree = ast.parse(source_code)
    lines = source_code.splitlines()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start_lineno = node.lineno - 1
            end_lineno = node.end_lineno
            return "\n".join(lines[:start_lineno] + lines[end_lineno:])

    return source_code


def remove_main_block(source_code: str) -> str:
    """
    Remove the if __name__ == "__main__": block from the source code.
    """
    tree = ast.parse(source_code)
    lines = source_code.splitlines()

    # Find the main block and note its line numbers
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):

                # Remove lines corresponding to this block
                start_lineno = node.lineno - 1
                end_lineno = node.end_lineno
                return "\n".join(lines[:start_lineno] + lines[end_lineno:])

    return source_code


def extract_top_level_functions_with_decorators_and_comments(
    code: str,
) -> List[Tuple[str, str]]:
    """
    Returns list of (function_name, source_segment) for top-level functions (excluding "main"),
    including decorators and contiguous preceding comments.
    """
    # Parse AST to get function nodes
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)

    # Precompute which line numbers have comment tokens
    comment_lines: Set[int] = set()
    lines = code.splitlines(keepends=True)  # preserve exact line content for prefix checks

    tokgen = tokenize.generate_tokens(io.StringIO(code).readline)  # yields (type, string, start, end, line)
    for tok_type, _, (srow, scol), _, _ in tokgen:
        if tok_type == tokenize.COMMENT:
            # everything before the comment on that line must be whitespace
            prefix = lines[srow - 1][:scol]
            if prefix.strip() == "":
                comment_lines.add(srow)

    functions = []

    for node in tree.body:  # only top-level
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == "main":
            continue

        # Determine the starting line: earliest decorator if present, else the def/async line
        if node.decorator_list:
            start_lineno = min(d.lineno for d in node.decorator_list)
        else:
            start_lineno = node.lineno

        # Extend upward to include contiguous comment lines (no intervening non-blank/non-comment)
        span_start = start_lineno
        curr = span_start - 1  # check line above; lines are 1-based
        while curr > 0:
            line_text = lines[curr - 1]
            if curr in comment_lines:
                span_start = curr
                curr -= 1
                continue
            if line_text.strip() == "":
                # blank line: include it and keep scanning upward
                span_start = curr
                curr -= 1
                continue
            break  # encountered code or something else; stop

        # Determine end line of the function definition including its body
        # Prefer end_lineno if available (Python 3.8+)
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            span_end = node.end_lineno
        else:
            # Fallback: get last lineno from the deepest child in body
            def _max_lineno(n):
                max_ln = getattr(n, "lineno", 0)
                for child in ast.iter_child_nodes(n):
                    ln = _max_lineno(child)
                    if ln > max_ln:
                        max_ln = ln
                return max_ln

            span_end = _max_lineno(node)

        # Slice the original source lines
        segment = "".join(lines[span_start - 1 : span_end])
        functions.append((node.name, segment))

    return functions


def split_code_and_output_into_sections(code: str, stdout: str) -> list[CodeSection]:
    """
    Converts a Python script and its output into a list of CodeSections.
    Pre-condition: The code in the main() function contains print statements that indicate section names, e.g., `print("Section: <section name>")`.
    """
    # This will hold all top-level code and by default all function definitions.
    # Functions will later be moved to more relevant sections if needed.
    # The first step is to remove both the if __name__ == "__main__": block and the main function
    top_level_code = remove_main_block(remove_function(code, "main"))

    main_function_body = extract_function_body(code, "main")
    functions = extract_top_level_functions_with_decorators_and_comments(top_level_code)

    # Split the main function body into sections based on print("Section: <section name>") code
    main_fn_top_level_section, main_fn_sections, known_section_names = (
        split_code_sections(main_function_body) if main_function_body else (None, [], [])
    )

    # Split the output into sections based on "Section: " headers
    output_top_level_section, output_sections = split_output_sections(stdout, known_section_names)

    # Merge code and outputs into code sections
    result_sections: list[CodeSection] = []
    for output_section, code_section in zip_longest(output_sections, main_fn_sections):
        name = None
        if code_section is not None:
            # If code section is available, extract the section name from it
            name = extract_first_section_name_from_code(code_section)
        elif output_section:
            # If only output section is available, extract the section name from it
            name = extract_first_section_name_from_output(output_section)
        comments, cleaned_code = (
            extract_comment_under_first_print(code_section) if code_section is not None else (None, None)
        )
        # Strip whitespaces for the cell
        if cleaned_code is not None:
            cleaned_code = cleaned_code.strip()
        result_sections.append(CodeSection(name=name, code=cleaned_code, comments=comments, output=output_section))

    # Small optimization: move function definitions to the sections where they are first called
    # TODO: this doesn't handle nested function references, e.g., fn A calls fn B which calls fn C
    # currently will not move C to the section where A is called
    for name, segment in functions:
        for section in result_sections:
            if section["code"] and is_function_called(section["code"], name):
                section["code"] = segment.strip() + "\n\n" + section["code"].lstrip()
                top_level_code = top_level_code.replace(segment, "")
                break

    # Inject the top-level code at the beginning of the sections
    top_level_code = (
        top_level_code.rstrip() + "\n\n" + main_fn_top_level_section.lstrip()
        if main_fn_top_level_section
        else top_level_code
    )
    result_sections.insert(
        0,
        CodeSection(
            name=None,
            code=top_level_code,
            comments=None,
            output=output_top_level_section,
        ),
    )

    return result_sections
