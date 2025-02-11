import shutil

def read_model_list(file_path):
    with open(file_path, 'r') as file:
        models = file.readlines()
    return [model.strip() for model in models]

model_list = read_model_list('model_list')
for model_name in model_list:
    model_name =  f"{model_name}-16k"
    env_path =  f'../envs/{model_name}.env'
    shutil.copy('.env',env_path)
    with open(env_path, 'a') as file:
        file.write(f"\nlitellm_chat_model=ollama/{model_name}\n")
    print(f"Creating {model_name}.env")