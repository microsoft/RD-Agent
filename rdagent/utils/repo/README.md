# RepoAnalyzer

RepoAnalyzer is a Python utility for analyzing and summarizing the contents of a Python repository. It provides a high-level overview of the repository structure, including a tree-like representation of the directory structure and details about files, classes, and functions.

## Features

- Generate a tree-like structure of the repository
- Summarize an entire repository
- Adjust verbosity levels for summaries
- Extract content from specific files
- Analyze Python files for classes and functions


## Usage

### Basic Usage

```python
from repo_utils import RepoAnalyzer

# Initialize the RepoAnalyzer with the path to your repository
repo_analyzer = RepoAnalyzer("/path/to/your/repo")

# Generate a summary of the repository
summary = repo_analyzer.summarize_repo()
print(summary)

# Extract content from specific files
highlighted_content = repo_analyzer.highlight(["file1.py", "file2.py"])
print(highlighted_content)
```

### Adjusting Verbosity Levels

You can adjust the verbosity of the summary using the following parameters:

- `verbose_level`: Controls the overall detail level of the summary
  - 0: Minimal (file names only)
  - 1: Default (file info, class names, function names)
  - 2+: Detailed (includes method details within classes)
- `doc_str_level`: Controls the inclusion of docstrings (0-2)
- `sign_level`: Controls the inclusion of function signatures (0-2)

Example:

```python
detailed_summary = repo_analyzer.summarize_repo(verbose_level=2, doc_str_level=1, sign_level=1)
print(detailed_summary)
```

## Example Output

### Repository Summary

```
Workspace Summary for my_project
========================================

Repository Structure:
my_project/
├── main.py
├── utils/
│   ├── helper.py
│   └── config.py
├── models/
│   ├── model_a.py
│   └── model_b.py

This workspace contains 5 Python files.

File 1 of 5:
File: main.py
----------------------------------------
This file contains 1 class and 2 top-level functions.

Class: MainApp
  Description: Main application class for the project.
  This class has 3 methods.

Function: setup_logging
  Accepts parameters: log_level
  Purpose: Configure the logging for the application.

Function: main
  Purpose: Entry point of the application.

...
```

### File Highlight

```python
highlighted_content = repo_analyzer.highlight(["main.py"])
print(highlighted_content["main.py"])
```

This will print the entire content of the `main.py` file.

## Key Components

### RepoAnalyzer Class

The main class that provides the functionality for analyzing repositories.

#### Methods:

- `summarize_repo(verbose_level=1, doc_str_level=1, sign_level=1)`: Generates a comprehensive summary of the repository, including a tree-like structure.
- `highlight(file_names)`: Extracts and returns the content of specified files.

### Tree-like Structure

The summary now includes a visual representation of the repository's directory structure, making it easier to understand the overall organization of the project.