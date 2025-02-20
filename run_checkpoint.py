import argparse
import os
import shutil
import subprocess
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.exp.researcher.utils import get_valid_submission

def run_checkpoint_for_single_loop(log_path, loop, checkpoint):
    command = [
        "dotenv",
        "run",
        "--",
        "env",
        f"LOG_TRACE_PATH={log_path}",
        "python",
        "rdagent/app/data_science/loop.py",
        "--path",
        checkpoint,
        "--loop_n",
        "2",
        ">",
        f"{log_path}_{loop}.log",
        "2>&1"
    ]
    try:
        # Run the command and wait for it to finish
        subprocess.run(" ".join(command), shell=True, check=True)
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

def process_single_competition(f, LOG_FOLDER, OUTPUT_FOLDER):
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
            status = run_checkpoint_for_single_loop(dst, first_loop, checkpoint)
            if status:
                loop_info['first_loop'] = first_loop
        if last_loop != -1 and last_loop != first_loop:
            checkpoint = f"{dst}/__session__/{last_loop}"
            checkpoint = f"{checkpoint}/{get_last_step(checkpoint)}"
            status = run_checkpoint_for_single_loop(dst, last_loop, checkpoint)
            if status:
                loop_info['last_loop'] = last_loop

        data.append({f.name: loop_info})

        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

def main():
    args = arg_parser()
    LOG_FOLDER = args.log_folder
    OUTPUT_FOLDER = args.output_folder
    num_process = args.num_process

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    with ThreadPoolExecutor(max_workers=num_process) as executor:
        futures = [executor.submit(process_single_competition, f, LOG_FOLDER, OUTPUT_FOLDER) for f in os.scandir(LOG_FOLDER)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing folder: {e}")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="./log1", help="The folder that store the log of baseline.")
    parser.add_argument("--output_folder", type=str, default="./log2", help="The folder that store the log of researcher.")
    parser.add_argument("--num_process", type=int, default=4, help="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()