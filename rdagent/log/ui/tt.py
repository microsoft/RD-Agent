from rdagent.utils.env import MLEBDockerEnv, DockerEnv, MLEBDockerConf
from pathlib import Path
from rdagent.log.storage import FileStorage
from rdagent.app.data_science.conf import DS_RD_SETTING

mle_de_conf = MLEBDockerConf()
mle_de_conf.extra_volumes = {
    f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data",
}
de = DockerEnv(conf=mle_de_conf)
de.prepare()

def save_grade_info(log_trace_path: str):
    for msg in FileStorage(log_trace_path).iter_msg():
        if "competition" in msg.tag:
            competition = msg.content

        if "running" in msg.tag:
            grade_output = msg.content.experiment_workspace.execute(env=de, entry=f"mlebench grade-sample submission.csv {competition} --data-dir /mle/data")
            (msg.content.experiment_workspace.workspace_path / "mle_score.txt").write_text(grade_output, encoding="utf-8")

if __name__ == "__main__":
    save_grade_info("log_trace_path")