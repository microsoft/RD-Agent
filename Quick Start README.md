# R&D Agent Project

> This repository has been populated with an initial template to help you get started. Please make sure to update the content to provide a great experience for community-building.

## Motivation
R&D Agent: Focusing on automating the most core and valuable parts of the industrial R&D process.

Core method: Evolving.

## Development

### Create a Conda Environment
- Create a new conda environment with Python 3.10:
  ```sh
  conda create -n rdagent python=3.10
  ```
- Activate the environment:
  ```sh
  conda activate rdagent
  ```

### Run Make Files
- **Navigate to the directory containing the MakeFile** and set up the development environment:
  ```sh
  make dev
  ```

### Install Pytorch
- Install Pytorch and related libraries:
  ```sh
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip3 install torch_geometric
  ```

### Environment Configuration
- Place the `.env` file (available in OneNote) in the same directory as the `.env.example` file.
- Export each variable in the `.env` file:
  ```sh
  export $(grep -v '^#' .env | xargs)
  ```

### Azure Configuration
- Install Azure CLI:
  ```sh
  curl -L https://aka.ms/InstallAzureCli | bash
  ```
- Log in to Azure:
  ```sh
  (your az root) az login --use-device-code
  ```
- Provide the code to your mentor and ask him to log in if needed.

- `exit` and re-login to your environment (this step may not be necessary).

### Run the Application
- Ensure you have the necessary financial report and factor implementation source data. You may need to copy the data from a remote location. For example:
  ```sh
  scp xuyang1@10.150.242.102:/home/xuyang1/workspace/fincov2/git_ignore_folder/factor_implementation_source_data /home/finco/v-yuanteli/rdagent/RD-Agent/git_ignore_folder/
  ```

- Update the `.env` file to set the `FILE_BASED_EXECUTION_DATA_FOLDER` variable:
  ```sh
  FILE_BASED_EXECUTION_DATA_FOLDER=/home/finco/v-yuanteli/rdagent/RD-Agent/git_ignore_folder/factor_implementation_source_data
  ```

- Run the factor extraction and implementation script:
  ```sh
  python rdagent/app/factor_extraction_and_implementation/factor_extract_and_implement.py
  ```
