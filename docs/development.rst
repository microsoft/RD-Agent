=========================
For Development
=========================

title1
=========================

content1

Code Structure
=========================

.. code-block:: text

    📂 src
    ➥ 📂 <project name>: avoid namespace conflict
      ➥ 📁 core
      ➥ 📁 component A
      ➥ 📁 component B
      ➥ 📁 component C
      ➥ 📂 app
        ➥ 📁 scenario1
        ➥ 📁 scenario2
    ➥ 📁 scripts

.. list-table::
   :header-rows: 1

   * - Folder Name
     - Description
   * - 📁 core
     - The core framework of the system. All classes should be abstract and usually can't be used directly.
   * - 📁 component X
     - Useful components that can be used by others (e.g., scenarios). Many subclasses of core classes are located here.
   * - 📁 app
     - Applications for specific scenarios (usually built based on components). Removing any of them does not affect the system's completeness or other scenarios.
   * - 📁 scripts
     - Quick and dirty things. These are candidates for core, components, and apps.


title3
=========================

content3
