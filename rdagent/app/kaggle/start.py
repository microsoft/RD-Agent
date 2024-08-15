from rdagent.utils.env import DockerEnv, KaggleConf

from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile


api = KaggleApi()
api.authenticate()

competition_name = "playground-series-s4e8"

# download data
local_data_path = f"/data/userdata/share/kaggle/{competition_name}"
api.competition_download_files(competition_name, path=local_data_path)
# unzip data
with zipfile.ZipFile(f"{local_data_path}/{competition_name}.zip", 'r') as zip_ref:
    zip_ref.extractall(local_data_path)

kaggle_conf = KaggleConf()
# mount kaggle input data
kaggle_conf.extra_volumes = {local_data_path: f"/kaggle/input/{competition_name}"}

kaggle_env = DockerEnv(kaggle_conf)

# use kaggle container to run the code
logs = kaggle_env.run(entry="ls /kaggle/input")
