# Tools Directory

This directory provides scripts to run experiments with different environment configurations, collect results, and demonstrate usage through an example script.

## Directory Structure

```
scripts/exp/tools/
├── run_envs.sh       # Script for running experiments
├── collect.py        # Results collection and summary
├── test_system.sh    # Usage script for rdagent kaggle loop
└── README.md         # This documentation
```

## Tools Overview

1. **run_envs.sh**: Executes experiments with different environment configurations in parallel.
2. **collect.py**: Collects and summarizes experiment results into a single file.
3. **test_system.sh**: Demonstrates how to use the above tools together for experiment execution and result collection (for rdagent kaggle loop).

## Getting Started

### Prerequisites

Place your `.env` files in the desired directory for environment configurations.

## Usage

### 1. Running Experiments with Different Environments

The `run_envs.sh` script allows running a command with multiple environment configurations in parallel.

**Command Syntax:**

```bash
./run_envs.sh -d <dir_to_.envfiles> -j <number_of_parallel_processes> -- <command>
```

**Example Usage:**

- Basic example:

   ```bash
   ./run_envs.sh -d env_files -j 1 -- echo "Hello"
   ```

- Practical example (running the kaggle loop file):

   ```bash
   dotenv run -- ./run_envs.sh -d RD-Agent/scripts/exp/ablation/env -j 1 -- python RD-Agent/rdagent/app/kaggle/loop.py
   ```

**Explanation:**

| Option      | Description                                                  |
|-------------|--------------------------------------------------------------|
| `-d`       | Specifies the directory containing `.env` files.            |
| `-j`       | Number of parallel processes to run (e.g., 1 for sequential execution). |
| `--`       | Separates script options from the command to execute.       |
| `<command>`| The command to execute with the environment variables loaded.|

### 2. Collecting Results

The `collect.py` script processes logs and generates a summary JSON file.

**Command Syntax:**

```bash
python collect.py --log_path <path_to_logs> --output_name <summary_filename>
```

**Example Usage:**

Collect results from logs:

```bash
python collect.py --log_path logs --output_name summary.json
```

**Explanation:**

| Option          | Description                                                  |
|-----------------|--------------------------------------------------------------|
| `--log_path`   | Required. Specifies the directory containing experiment logs.|
| `--output_name`| Optional. The name of the output summary file (default: summary.json). |

### 3. Example Workflow (for rdagent kaggle loop)

Use the `test_system.sh` script to demonstrate a complete workflow.

**Steps:**

1. Run the test system:

   ```bash
   ./scripts/exp/tools/test_system.sh
   ```

   This will:
   1. Load environment configurations from `.env` files.
   2. Execute experiments using the configurations.

2. Find your logs in the `logs` directory.

3. Use the `collect.py` script to summarize results:

   ```bash
   python collect.py --log_path logs --output_name summary.json
   ```

## Create Your Own Workflow

- Create the ablation environments under a specified folder.
- Revise the `test_system.sh` template to adjust the path and relevant commands for execution.
- Run `test_system.sh` to execute the environments through different configurations.
- Keep track of your log path and use `collect.py` to collect the results at scale.

## Notes

- Scale parallel processes as needed using the `-j` parameter.
- Avoid errors by ensuring `.env` files are correctly formatted.
- Modify `test_system.sh` to meet your project's specific needs.
- Add other metrics of interest in `collect.py` to summarize automatically.

For further assistance, refer to the comments within the scripts or reach out to the development team.