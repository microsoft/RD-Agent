===============================
Framework Design & Components
===============================

Framework & Components
=========================

- TODO: Components & Feature Level
![image](https://github.com/user-attachments/assets/c622704c-377a-4361-b956-c1eb9cf6a736)

- Class Level Figure
![image](https://github.com/user-attachments/assets/60cc2712-c32a-4492-a137-8aec59cdc66e)

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

