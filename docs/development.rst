=========================
For Development
=========================

If you want to try the latest version or contribute to RD-Agent. You can install it from the source and follow the commands in this page.

   .. code-block:: bash

      git clone https://github.com/microsoft/RD-Agent


🔧Prepare for development
=========================

- Set up the development environment.

   .. code-block:: bash

      make dev

- Run linting and checking.

   .. code-block:: bash

      make lint


- Some linting issues can be fixed automatically. We have added a command in the Makefile for easy use.

   .. code-block:: bash

      make auto-lint



Code Structure
=========================

.. code-block:: text

    📂 src
    ➥ 📂 <project name>: avoid namespace conflict
      ➥ 📁 core
      ➥ 📁 components/A
      ➥ 📁 components/B
      ➥ 📁 components/C
      ➥ 📁 scenarios/X
      ➥ 📁 scenarios/Y
      ➥ 📂 app
    ➥ 📁 scripts

.. list-table::
   :header-rows: 1

   * - Folder Name
     - Description
   * - 📁 core
     - The core framework of the system. All classes should be abstract and usually can't be used directly.
   * - 📁 component/A
     - Useful components that can be used by others (e.g., scenarios). Many subclasses of core classes are located here.
   * - 📁 scenarios/X
     - Concrete features for specific scenarios (usually built based on components or core). These modules are often unreusable across scenarios.
   * - 📁 app
     - Applications for specific scenarios (usually built based on components or scenarios). Removing any of them does not affect the system's completeness or other scenarios.
   * - 📁 scripts
     - Quick and dirty things. These are candidates for core, components, scenarios, and apps.



Conventions
===========


File Naming Convention
----------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - `conf.py`
     - The configuration for the module, app, and project.

.. <!-- TODO: renaming files -->


Security boundaries for generated code and artifacts
====================================================

RD-Agent executes generated code and processes files produced by external
tools. Treat filenames, archive members, subprocess output, competition names,
and environment names as untrusted input even when the surrounding workflow is
started by a trusted operator.

Workspace file operations
-------------------------

Use :meth:`FBWorkspace.inject_files <rdagent.core.experiment.FBWorkspace.inject_files>`
and :meth:`FBWorkspace.remove_files <rdagent.core.experiment.FBWorkspace.remove_files>`
for files owned by an experiment workspace. These methods accept relative paths
inside ``workspace_path``. Parent traversal, absolute paths, and paths that
escape through a symbolic link are rejected before a file is written or
deleted.

Do not construct an unchecked path from a generated filename and then call
``write_text()``, ``unlink()``, or another filesystem operation directly. New
workspace APIs should apply the same resolved-path containment rule to every
write and delete branch.

Archive extraction
------------------

Use ``rdagent.utils.archive.safe_extract_zip`` and
``rdagent.utils.archive.safe_extract_tar`` for archives that may contain
externally supplied entries. The helpers reject:

* absolute and parent-traversal member paths;
* symbolic links and hard links;
* device nodes and other special file types; and
* archives containing more than 10,000 members by default.

Avoid ``ZipFile.extractall()`` and ``TarFile.extractall()`` in these paths. If a
workflow legitimately needs links, special files, or a larger archive, handle
that input in a separately reviewed trusted-data path rather than weakening the
shared helpers.

Commands and parsed process output
----------------------------------

Pass subprocess arguments as a list with ``shell=False`` whenever any argument
can vary. Conda environment names are limited to letters, digits, ``_``, ``-``,
and ``.``, and Python versions must be numeric dotted versions. Kaggle
competition identifiers use the corresponding competition-slug validator.

Never use ``eval()`` to parse subprocess or container output. Score output must
be finite numeric JSON. Legacy Python dictionary-shaped training metrics may be
parsed with ``ast.literal_eval()`` only when JSON cannot be used.
