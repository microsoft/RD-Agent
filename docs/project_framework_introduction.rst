===============================
Framework Design & Components
===============================

Framework & Components
=========================

- TODO: Components & Feature Level

- Class Level Figure

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

