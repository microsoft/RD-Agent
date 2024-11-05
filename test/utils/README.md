# 🐳 Run Docker & Qlib
---

## 📄 Description
This guide explains how to run the Qlib Docker test file located at `test/utils/test_env.py` in the RD-Agent repository.

---

## 🚀 Running Instructions

### 1. Install the required Python libraries
- Ensure that the `docker` Python library is installed:
    ```sh
    pip install docker
    ```

### 2. Run the test script
- Execute the test script to verify the Docker environment setup:
    ```sh
    python test/utils/test_env.py
    ```

### Troubleshooting
- **PermissionError: [Errno 13] Permission denied.**
    > This error occurs when the current user does not have the necessary permissions to access the Docker socket. To resolve this issue, follow these steps:

1. **Add the current user to the `docker` group**
Docker requires root or `docker` group user permissions to access the Docker socket. Add the current user to the `docker` group:
    ```sh
    sudo usermod -aG docker $USER
    ```

2. **Refresh group changes**
To apply the group changes, log out and log back in, or use the following command:
    ```sh
    newgrp docker
    ```

3. **Verify Docker access**
Run the following command to ensure that Docker can be accessed:
    ```sh
    docker run hello-world
    ```

4. **Rerun the test script**
    After completing these steps, rerun the test script:
    ```sh
    python test/utils/test_env.py
    ```
---
## 🛠️ Detailed Qlib Docker Function Framework

Here, we provide an overview of the specific functions within the Qlib Docker framework, their purposes, and examples of how to call them.

### QTDockerEnv Class in `env.py`

The `QTDockerEnv` class is responsible for setting up and running Docker environments for Qlib experiments. 

#### Methods:

1. **prepare()**
   - **Purpose**: Prepares the Docker environment for running experiments. This includes building the Docker image if necessary.
   - **Example**:
     ```python
     qtde = QTDockerEnv()
     qtde.prepare()
     ```

2. **run(local_path: str, entry: str) -> str**
   - **Purpose**: Runs a specified entry point (e.g., a configuration file) in the prepared Docker environment.
   - **Parameters**:
     - `local_path`: Path to the local directory to mount into the Docker container.
     - `entry`: Command or entry point to run inside the Docker container.
   - **Returns**: The stdout output from the Docker container.
   - **Example**:
     ```python
     result = qtde.run(local_path="/path/to/env_tpl", entry="qrun conf.yaml")
     ```
---
### 📊 Expected Output

Upon successful execution, the test script will produce analysis results of benchmark returns and various risk metrics. The expected output should be similar to:

```
'The following are analysis results of benchmark return (1 day).'
risk
mean               0.000477
std                0.012295
annualized_return  0.113561
information_ratio  0.598699
max_drawdown      -0.370479

'The following are analysis results of the excess return without cost (1 day).'
risk
mean               0.000530
std                0.005718
annualized_return  0.126029
information_ratio  1.428574
max_drawdown      -0.072310

'The following are analysis results of the excess return with cost (1 day).'
risk
mean               0.000339
std                0.005717
annualized_return  0.080654
information_ratio  0.914486
max_drawdown      -0.086083

'The following are analysis results of indicators (1 day).'
value
ffr    1.0
pa     0.0
pos    0.0
```

By following these steps and using the provided functions, you should be able to run the Qlib Docker tests and obtain the expected analysis results.