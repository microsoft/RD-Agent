# Introduction

!!!!!This dev container is not for public development!!!!!!
!!!!!Please don't use it if you are just a public open-source user.!!!!!!

# Steps to run the dev container (for internal use only)

Prerequisites(this is the reason why this dev container is not for public use):

- Make sure you have the `rdagentappregistry.azurecr.io/rd-agent-mle:20250623` image locally & DevContainer is installed in your IDE
- The kaggle dataset is located at `/home/shared/RD-Agent/kaggle`

1. Open the project and select "Open In DevContainer"
2. Set up your Kaggle Key (do not share this; other internal URLs are hardcoded in the config files)

```bash
export KAGGLE_USERNAME=
export KAGGLE_KEY=
```

3. Run: python rdagent/app/data_science/loop.py --competition nomad2018-predict-transparent-conductors


# Additional Notes
- Please install and use this Dev Container in VS Code.
- You **must open VS Code remotely and enter the `RD-Agent` directory before running the DevContainer configuration (`.devcontainer/devcontainer.json`)**. Otherwise, the workspace and path mappings will not work as expected.
- To open the DevContainer correctly in VS Code:
  1. Remotely connect to the machine and open the `RD-Agent` folder in VS Code.
  2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac), type and select **"Dev Containers: Reopen in Container"**.



# How to grade your submission in the DevContainer

1. save your submission file in `./sumission.csv`

2. Run evaluation
DS_COMPETITION=<your competition name>
conda run -n mlebench  mlebench grade-sample submission.csv $DS_COMPETITION --data-dir /tmp/kaggle/zip_files/