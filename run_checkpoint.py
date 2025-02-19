import os, shutil, subprocess
from scripts.exp.researcher.utils import get_first_valid_submission

LOG_FOLDER = "./log" # baseline - {competition_id}_baseline
OUTPUT_FOLDER = "./log2" # researcher - {competition_id}_researcher
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for f in os.scandir(LOG_FOLDER):
    if f.is_dir() and "baseline" in f.name:
        src = f"{LOG_FOLDER}/{f.name}"
        dst = f"{OUTPUT_FOLDER}/{f.name.replace('baseline', 'researcher')}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        valid_loop = get_first_valid_submission(dst)
        checkpoint = f"{dst}/__session__/{valid_loop}/4_record"

        command = [
            "dotenv",
            "run",
            "--",
            "env",
            f"LOG_TRACE_PATH={dst}",
            "python",
            "rdagent/app/data_science/loop.py",
            "--path",
            checkpoint,
            "--loop_n",
            "2"
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Successfully executed command for checkpoint: {checkpoint}")
        except subprocess.CalledProcessError as e:
            print(f"Error while running command for checkpoint: {checkpoint}")
            print(e)

