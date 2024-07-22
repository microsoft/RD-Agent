===============================
Framework Design & Components
===============================

title1
=========================

content1

Framework & Components
=========================

content2

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



Conventions
===========

## File naming convention

| Name      | Description       |
| --        | --                |
| `conf.py` | The configuration for the module & app & project  | 

<!-- TODO: renaming files -->
