===============================
Framework Design & Components
===============================

Framework & Components
=========================

.. NOTE: This depends on the correctness of `c-v` of github.

.. image:: https://github.com/user-attachments/assets/98fce923-77ab-4982-93c8-a7a01aece766
    :alt: Components & Feature Level

The image above shows the overall framework of RDAgent.


.. image:: https://github.com/user-attachments/assets/60cc2712-c32a-4492-a137-8aec59cdc66e
    :alt: Class Level Figure

For those interested in the detailed code, the figure above illustrates the main classes and aligns them with the workflow.


Detailed Design
=========================


Configuration
-------------

You can manually source the `.env` file in your shell before running the Python script:
Most of the workflow are controlled by the environment variables.
```sh
# Export each variable in the .env file; Please note that it is different from `source .env` without export
export $(grep -v '^#' .env | xargs)
# Run the Python script
python your_script.py
```

