from rdagent.utils.env import DockerEnv, MLEBDockerConf
from pathlib import Path
from rdagent.log.storage import FileStorage
import fire

mle_de_conf = MLEBDockerConf()
mle_de_conf.extra_volumes = {
    f"/data/userdata/share/mle_kaggle/zip_files": "/mle/data",
}
de = DockerEnv(conf=mle_de_conf)
de.prepare()

def save_grade_info(log_trace_path):
    for msg in FileStorage(log_trace_path).iter_msg():
        if "competition" in msg.tag:
            competition = msg.content

        if "running" in msg.tag:
            msg.content.experiment_workspace.execute(env=de, entry=f"bash -c 'mlebench grade-sample submission.csv {competition} --data-dir /mle/data > mle_score.txt 2>&1'")
            msg.content.experiment_workspace.execute(env=de, entry="chmod 777 mle_score.txt")

def save_all_grade_info(log_folder):
    for log_trace_path in log_folder.iterdir():
        save_grade_info(log_trace_path)

if __name__ == "__main__":
    fire.Fire({
        'run': save_all_grade_info
    })