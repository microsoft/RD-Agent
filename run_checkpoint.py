import os, shutil, subprocess, json, re
from scripts.exp.researcher.utils import get_valid_submission

LOG_FOLDER = "./log1" # baseline - {competition_id}_baseline
OUTPUT_FOLDER = "./log2" # researcher - {competition_id}_researcher
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def run_checkpoint_for_single_loop(checkpoint):
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
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error while running command for checkpoint: {checkpoint}")
        print(e)
        return False

def get_last_step(session_path):
    steps = os.listdir(session_path)
    idx, step = -1, ""
    for s in steps:
        cur_idx = int(re.findall(r'\d+', s)[0])
        if cur_idx > idx:
            idx = cur_idx
            step = s
    return step

for f in os.scandir(LOG_FOLDER):
    if f.is_dir() and "baseline" in f.name:
        src = f"{LOG_FOLDER}/{f.name}"
        dst = f"{OUTPUT_FOLDER}/{f.name.replace('baseline', 'researcher')}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        output_path = f"{OUTPUT_FOLDER}/loop.json"
        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                data = json.load(file)
        else:
            data = []
        loop_info = {}


        first_loop, last_loop = get_valid_submission(dst)
        print(f"Competition: {f.name} | First Loop: {first_loop} | Last Loop: {last_loop}")
        if first_loop != -1:
            checkpoint = f"{dst}/__session__/{first_loop}"
            checkpoint = f"{checkpoint}/{get_last_step(checkpoint)}"
            status = run_checkpoint_for_single_loop(checkpoint)
            if status:
                loop_info['first_loop'] = first_loop
        if last_loop != -1 and last_loop != first_loop:
            checkpoint = f"{dst}/__session__/{last_loop}"
            checkpoint = f"{checkpoint}/{get_last_step(checkpoint)}"
            status = run_checkpoint_for_single_loop(checkpoint)     
            if status:
                loop_info['first_loop'] = last_loop

        data.append({f.name: loop_info})

        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

