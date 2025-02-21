import argparse
import os
import shutil
import subprocess
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.exp.researcher.utils import get_valid_submission

def get_last_step(session_path):
    steps = os.listdir(session_path)
    idx, step = -1, ""
    for s in steps:
        cur_idx = int(re.findall(r'\d+', s)[0])
        if cur_idx > idx:
            idx = cur_idx
            step = s
    return step


def continue_checkpoint(loop_idx, src, dst, n_loops): 
    if loop_idx != -1: 
        path = f"{src}/__session__/{loop_idx}"
        path = f"{path}/{get_last_step(path)}"
        output_path = dst
        command = [
            "dotenv",
            "run",
            "--",
            "python",
            "rdagent/app/data_science/loop.py",
            "--path",
            path,
            "--output_path",
            output_path,
            "--loop_n",
            str(n_loops),
            ">",
            f"{output_path}_{loop_idx}.log",
            "2>&1"
        ]
        try:
            subprocess.run(" ".join(command), shell=True, check=True)
            print(f"**Success** | Path: {path} | Output Path: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"**Fail** | Path: {path} | Output Path: {output_path}")
            print(e)
            return False
    return False


def run_single_checkpoint(competition, path, output_path, n_loops):
    if competition.is_dir():
        src = f"{path}/{competition.name}"
        dst = f"{output_path}/{competition.name.replace('baseline', 'researcher')}"

        first_loop, last_loop = get_valid_submission(src)
        print(f"Competition: {competition.name} | First Loop: {first_loop} | Last Loop: {last_loop} | Output Path: {dst}")
        continue_checkpoint(first_loop, src, dst, n_loops)
        continue_checkpoint(last_loop, src, dst, n_loops)

def main():
    args = arg_parser()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with ThreadPoolExecutor(max_workers=args.n_process) as executor:
        futures = [executor.submit(run_single_checkpoint, f, args.path, args.output_path, args.n_loops) for f in os.scandir(args.path)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"**Fail** | {e}")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./log_baseline", help="The checkpoint path that store the logs and sessions of baseline.")
    parser.add_argument("--output_path", type=str, default="./log_researcher", help="The output path that save the logs and sessions.")
    parser.add_argument("--n_process", type=int, default=4, help="Number of jobs to run in parallel.")
    parser.add_argument("--n_loops", type=int, default=2, help="Number of loops to continue.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()