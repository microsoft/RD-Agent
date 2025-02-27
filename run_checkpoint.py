import argparse
import os
import shutil
import subprocess
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.exp.researcher.utils import get_last_step, get_loop_idx
from rdagent.app.data_science.loop import DataScienceRDLoop


def continue_checkpoint(loop_idx, src, dst, n_loops, loop_type): 
    if loop_idx != -1: 
        path = f"{src}/__session__/{loop_idx}"
        path = f"{path}/{get_last_step(path)}"
        output_path = dst
        command = [
            "timeout",
            "$((3*60*60))",
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
            f"{output_path}_{loop_type}_{loop_idx}.log",
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
        dst = f"{output_path}/{competition.name}"

        es_loop, ls_loop = get_loop_idx(src)
        print(f"Competition: {competition.name} | Early Stage: {es_loop} | Late Stage: {ls_loop}")
        continue_checkpoint(es_loop, src, dst, n_loops, "early")
        if es_loop != ls_loop: 
            continue_checkpoint(ls_loop, src, dst, n_loops, "late")


def main():
    args = arg_parser()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    competitions = [entry for entry in sorted(os.scandir(args.path), key=lambda e: e.name) if entry.is_dir()]
    if args.max_num:
        competitions = competitions[:args.max_num]

    with ThreadPoolExecutor(max_workers=args.n_process) as executor:
        futures = [executor.submit(run_single_checkpoint, f, args.path, args.output_path, args.n_loops) for f in competitions]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"**Fail** | {e}")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./log_checkpoint", help="The checkpoint path that store the logs and sessions of baseline.")
    parser.add_argument("--output_path", type=str, default="./log_researcher", help="The output path that save the logs and sessions.")
    parser.add_argument("--n_process", type=int, default=4, help="Number of jobs to run in parallel.")
    parser.add_argument("--n_loops", type=int, default=2, help="Number of loops to continue.")
    parser.add_argument("--max_num", type=int, default=None, help="Number of competitions to run in total.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()