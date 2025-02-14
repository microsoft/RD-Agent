import shutil

def read_model_list(file_path):
    with open(file_path, 'r') as file:
        models = file.readlines()
    return [model.strip() for model in models]
model_dict = {
    'DeepSeek-V3':'deepseek/ep-20250208201555-qp2g4',
    'DeepSeek-R1':'deepseek/ep-20250208191624-h8chn',
    'Claude-3.5-Sonnet':'anthropic/claude-3-5-sonnet-20240620'
}
for model_name in model_dict:
    env_path =  f'../envs/{model_name}.env'
    shutil.copy('.env',env_path)
    with open(env_path, 'a') as file:
        file.write(f"\nlitellm_chat_model={model_dict[model_name]}\n")
    print(f"Creating {model_name}.env")