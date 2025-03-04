import argparse
import os
import shutil
import subprocess
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.exp.researcher.utils import get_last_step, get_loop_idx
from rdagent.app.data_science.loop import DataScienceRDLoop


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./log_checkpoint", help="The checkpoint path that store the logs and sessions of baseline.")
    parser.add_argument("--output_path", type=str, default="./log_researcher", help="The output path that save the logs and sessions.")
    parser.add_argument("--n_process", type=int, default=4, help="Number of jobs to run in parallel.")
    parser.add_argument("--n_loop", type=int, default=2, help="Number of loops to continue.")
    parser.add_argument("--n_round", type=int, default=1, help="Number of rounds to run in total.")
    parser.add_argument("--max_num", type=int, default=None, help="Number of competitions to run in total.")
    
    args = parser.parse_args()
    return args


def get_loop_idx(log_trace_path):
    session_path = f"{log_trace_path}/__session__"
    es_loop = ls_loop = -1
    for loop in os.listdir(session_path):
        loop_idx = int(loop)
        session = f"{session_path}/{loop}"
        session = f"{session}/{get_last_step(session)}"
        kaggle_loop = DataScienceRDLoop.load(path=session)
        if kaggle_loop.trace.next_incomplete_component() is None: # all component are complete
            if loop_idx < es_loop or es_loop == -1:
                es_loop = loop_idx
            
        if loop_idx > ls_loop:
            ls_loop = loop_idx

    return es_loop, ls_loop


def get_last_step(session_path):
    steps = os.listdir(session_path)
    idx, step = -1, ""
    for s in steps:
        cur_idx = int(re.findall(r'\d+', s)[0])
        if cur_idx > idx:
            idx = cur_idx
            step = s
    return step


def continue_checkpoint(loop_idx, src, dst, n_loop, loop_type): 
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
            str(n_loop),
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


def run_single_checkpoint(competition, path, output_path, n_loop):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    src = f"{path}/{competition.name}"
    dst = f"{output_path}/{competition.name}"

    es_loop, ls_loop = get_loop_idx(src)
    print(f"Competition: {competition.name} | Early Stage: {es_loop} | Late Stage: {ls_loop} | Output Path: {output_path}")
    continue_checkpoint(es_loop, src, dst, n_loop, "early")
    if es_loop != ls_loop: 
        continue_checkpoint(ls_loop, src, dst, n_loop, "late")


def main():
    args = arg_parser()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    competitions = [entry for entry in sorted(os.scandir(args.path), key=lambda e: e.name) if entry.is_dir()]
    n_competitions = len(competitions)
    competitions = competitions * args.n_round
    if args.max_num:
        competitions = competitions[:args.max_num]

    with ThreadPoolExecutor(max_workers=args.n_process) as executor:
        futures = [executor.submit(run_single_checkpoint, 
                                   competition = competition, 
                                   path = args.path, 
                                   output_path = f"{args.output_path}/round_{i // n_competitions}", 
                                   n_loop = args.n_loop) 
                                   for i, competition in enumerate(competitions)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"**Fail** | {e}")


if __name__ == "__main__":
    main()