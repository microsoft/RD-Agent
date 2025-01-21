from rdagent.utils.env import MLEBDockerEnv
from pathlib import Path

mleb_env = MLEBDockerEnv()
mleb_env.prepare()
mleb_env.run(
    f"mlebench prepare -c aerial-cactus-identification --data-dir ./zip_files --skip-verification",
    local_path="/data/userdata/share/mle_kaggle",
    running_extra_volume={str(Path("~/.kaggle").expanduser().absolute()): "/root/.kaggle"},
)
print('hh')