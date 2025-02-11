import argparse
import os
import glob
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv

num_of_workers = 3
env_dir = '../envs'
command = ['/data/userdata/v-yihuachen/anaconda3/envs/rdagent-dev/bin/python',
          '/data/userdata/v-yihuachen/repo/RD-Agent/rdagent/app/benchmark/factor/eval.py']

def signal_handler(signum, frame):
    print("\n收到中断信号，正在清理并退出...")
    sys.exit(0)

def run_command_with_env(env_file, command):
    try:
        model_name = env_file.split('/')[-1].rsplit('.', 1)[0]
        cmd = command.copy()  # 创建命令的副本，避免修改原始命令
        cmd.append(f'>{model_name}.log')
        # 加载环境变量文件中的环境变量
        load_dotenv(env_file)
        print(f"使用 {env_file} 运行命令: {' '.join(cmd)}")
        os.system(' '.join(cmd))
    except KeyboardInterrupt:
        print(f"\n进程被中断: {env_file}")
        return

def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 检查目录是否存在
        if not os.path.isdir(env_dir):
            print(f"目录 {env_dir} 不存在。")
            exit(1)

        # 查找目录中的所有 .env 文件
        env_files = glob.glob(os.path.join(env_dir, '*.env'))
        if not env_files:
            print(f"{env_dir} 中没有找到 .env 文件。")
            exit(1)

        # 打印将要使用的环境文件
        print("将使用以下 env 文件运行实验:")
        for env_file in env_files:
            print(env_file)

        # 并行运行命令，每个 .env 文件一个进程
        with ProcessPoolExecutor(max_workers=num_of_workers) as executor:
            futures = []
            for env_file in env_files:
                future = executor.submit(run_command_with_env, env_file, command)
                futures.append(future)
            
            # 等待所有任务完成或被中断
            for future in futures:
                try:
                    future.result()
                except KeyboardInterrupt:
                    print("\n正在终止所有运行中的任务...")
                    executor.shutdown(wait=False)
                    break
                except Exception as e:
                    print(f"任务执行出错: {e}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()