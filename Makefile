.PHONY: clean deepclean install init-qlib-env dev constraints black isort mypy ruff toml-sort lint pre-commit test-run test build upload docs-autobuild changelog docs-gen docs-mypy docs-coverage docs
#You can modify it according to your terminal
SHELL := /bin/bash

########################################################################################
# Variables
########################################################################################

# Determine whether to invoke pipenv based on CI environment variable and the availability of pipenv.
PIPRUN := $(shell [ "$$CI" != "true" ] && command -v pipenv > /dev/null 2>&1 && echo "pipenv run")

# Get the Python version in `major.minor` format, using the environment variable or the virtual environment if exists.
PYTHON_VERSION := $(shell echo $${PYTHON_VERSION:-$$(python -V 2>&1 | cut -d ' ' -f 2)} | cut -d '.' -f 1,2)

# Determine the constraints file based on the Python version.
CONSTRAINTS_FILE := constraints/$(PYTHON_VERSION).txt

# Documentation target directory, will be adapted to specific folder for readthedocs.
PUBLIC_DIR := $(shell [ "$$READTHEDOCS" = "True" ] && echo "$$READTHEDOCS_OUTPUT/html" || echo "public")

# URL and Path of changelog source code.
CHANGELOG_URL := $(shell echo $${CI_PAGES_URL:-https://microsoft.github.io/rdagent}/_sources/changelog.md.txt)
CHANGELOG_PATH := docs/changelog.md

########################################################################################
# Development Environment Management
########################################################################################

# Remove common intermediate files.
clean:
	-rm -rf \
		$(PUBLIC_DIR) \
		.coverage \
		.mypy_cache \
		.pytest_cache \
		.ruff_cache \
		Pipfile* \
		coverage.xml \
		dist \
		release-notes.md
	find . -name '*.egg-info' -print0 | xargs -0 rm -rf
	find . -name '*.pyc' -print0 | xargs -0 rm -f
	find . -name '*.swp' -print0 | xargs -0 rm -f
	find . -name '.DS_Store' -print0 | xargs -0 rm -f
	find . -name '__pycache__' -print0 | xargs -0 rm -rf

# Remove pre-commit hook, virtual environment alongside itermediate files.
deepclean: clean
	if command -v pre-commit > /dev/null 2>&1; then pre-commit uninstall --hook-type pre-push; fi
	if command -v pipenv >/dev/null 2>&1 && pipenv --venv >/dev/null 2>&1; then pipenv --rm; fi

# Install the package in editable mode.
install:
	$(PIPRUN) pip install -e . -c $(CONSTRAINTS_FILE)

# Install the package in editable mode with specific optional dependencies.
dev-%:
	$(PIPRUN) pip install -e .[$*] -c $(CONSTRAINTS_FILE)

# Prepare the development environment.
# Build submodules.
# Install the pacakge in editable mode with all optional dependencies and pre-commit hook.
init-qlib-env:
	# note: You may need to install torch manually
	# todo: downgrade ruamel.yaml in pyqlib
	conda create -n qlibRDAgent python=3.8 -y
	@source $$(conda info --base)/etc/profile.d/conda.sh && conda activate qlibRDAgent && which pip && pip install pyqlib && pip install ruamel-yaml==0.17.21 && pip install torch==2.1.1 && pip install catboost==0.24.3 && conda deactivate

dev:
	$(PIPRUN) pip install -e .[docs,lint,package,test] -c $(CONSTRAINTS_FILE)
	if [ "$(CI)" != "true" ] && command -v pre-commit > /dev/null 2>&1; then pre-commit install --hook-type pre-push; fi

# Generate constraints for current Python version.
constraints: deepclean
	$(PIPRUN) --python $(PYTHON_VERSION) pip install --upgrade -e .[docs,lint,package,test]
	$(PIPRUN) pip freeze --exclude-editable > $(CONSTRAINTS_FILE)

########################################################################################
# Lint and pre-commit
########################################################################################

# Check lint with black.
black:
	$(PIPRUN) python -m black --check . --extend-exclude test/scripts --extend-exclude git_ignore_folder -l 120

# Check lint with isort.
isort:
	$(PIPRUN) python -m isort --check . -s git_ignore_folder -s test/scripts

# Check lint with mypy.
mypy:
	$(PIPRUN) python -m mypy . --exclude rdagent/scripts --exclude git_ignore_folder

# Check lint with ruff.
ruff:
	$(PIPRUN) ruff check .  --exclude rdagent/scripts,git_ignore_folder

# Check lint with toml-sort.
toml-sort:
	$(PIPRUN) toml-sort --check pyproject.toml

# Check lint with all linters.
lint: mypy ruff toml-sort

# Run pre-commit with autofix against all files.
pre-commit:
	pre-commit run --all-files

########################################################################################
# Test
########################################################################################

# Clean and run test with coverage.
test-run:
	$(PIPRUN) python -m coverage erase
	$(PIPRUN) python -m coverage run --concurrency=multiprocessing -m pytest --ignore test/scripts
	$(PIPRUN) python -m coverage combine

# Generate coverage report for terminal and xml.
test: test-run
	$(PIPRUN) python -m coverage report --fail-under 80
	$(PIPRUN) python -m coverage xml --fail-under 80

########################################################################################
# Package
########################################################################################

# Build the package.
build:
	$(PIPRUN) python -m build

# Upload the package.
upload:
	$(PIPRUN) python -m twine upload dist/*

########################################################################################
# Documentation
########################################################################################

# Generate documentation with auto build when changes happen.
docs-autobuild:
	$(PIPRUN) python -m sphinx_autobuild docs $(PUBLIC_DIR) \
		--watch README.md \
		--watch rdagent

# Generate changelog from git commits.
# NOTE(xuan.hu): Need to be run before document generation to take effect.
changelog:
	@if wget -q --spider $(CHANGELOG_URL); then \
		echo "Existing Changelog found at '$(CHANGELOG_URL)', download for incremental generation."; \
		wget -q -O $(CHANGELOG_PATH) $(CHANGELOG_URL); \
	fi
	$(PIPRUN) git-changelog -ETrio $(CHANGELOG_PATH) -c conventional -s build,chore,ci,docs,feat,fix,perf,refactor,revert,style,test

# Generate release notes from changelog.
release-notes:
	@$(PIPRUN) git-changelog --input $(CHANGELOG_PATH) --release-notes

# Build documentation only from rdagent.
docs-gen:
	$(PIPRUN) python -m sphinx.cmd.build docs $(PUBLIC_DIR)

# Generate mypy reports.
docs-mypy: docs-gen
	$(PIPRUN) python -m mypy rdagent test --exclude git_ignore_folder --exclude rdagent/scripts --html-report $(PUBLIC_DIR)/reports/mypy

# Generate html coverage reports with badge.
docs-coverage: test-run docs-gen
	$(PIPRUN) python -m coverage html -d $(PUBLIC_DIR)/reports/coverage --fail-under 80
	$(PIPRUN) bash scripts/generate-coverage-badge.sh $(PUBLIC_DIR)/_static/badges

# Generate all documentation with reports.
docs: changelog docs-gen docs-mypy docs-coverage


########################################################################################
# End
########################################################################################
