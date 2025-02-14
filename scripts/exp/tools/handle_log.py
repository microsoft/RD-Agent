import shutil,os,subprocess  

from dotenv import load_dotenv
command = ['/data/userdata/v-yihuachen/anaconda3/envs/rdagent-dev/bin/python','/data/userdata/v-yihuachen/repo/RD-Agent/rdagent/app/benchmark/factor/analysis.py' ]
env_file = '/data/userdata/v-yihuachen/repo/RD-Agent/rdagent/app/benchmark/factor/.env'
def read_model_list(file_path):
    with open(file_path, 'r') as file:
        models = file.readlines()
    return [model.strip() for model in models]

def analysis_log_to_result(log_name,pkl_path):
    command_iter = command.copy()
    command_iter.append(pkl_path)
    # command_iter.append( '>>result.txt')
    # 加载环境变量文件中的环境变量
    load_dotenv(env_file)
    # os.system(' '.join(['echo',f'Handling  {log_path}','>>result.txt']))
    # os.system(' '.join(command_iter))
    command_str = f"{' '.join(command)} {pkl_path} "
    # 执行echo命令
    os.system(f'echo "Handling {log_name}\nAnalysing {pkl_path}" >> result.txt')
    
    # 执行主命令
    os.system(command_str)
    
    
model_list = read_model_list('model_list')
# model_list = model_list[:10]
# model_list = model_list[10:]
for model_name in model_list:
    model_name =  f"{model_name}-16k"
    log_name = f"{model_name}.log"
    log_backup_path =  f'./logs_backup/{model_name}.log'
    # shutil.copy(log_name,log_backup_path)    # 备份log，不需要每次运行
    print(f"Handling and backup {model_name}.log")

    with open(log_name, 'r') as file:
        log = file.readlines()
        # 读取log 最后一行
        last_line = log[-1]
        print(last_line)
        # 处理log
        # Logging object in /data/userdata/v-yihuachen/repo/RD-Agent/scripts/exp/tools/log/2025-02-12_06-38-19-283990/491161/2025-02-12_09-13-29-822543.pkl
        # 把这行的pkl文件路径提取出来
        pkl_path = last_line.split(' ')[-1]
        print(pkl_path)
        # 执行python 代码 并把结果存到一个文件中
        analysis_log_to_result(log_name,pkl_path)