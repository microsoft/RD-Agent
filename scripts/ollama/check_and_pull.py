import subprocess

def read_model_list(file_path):
    with open(file_path, 'r') as file:
        models = file.readlines()
    return [model.strip() for model in models]

def pull_models(models):
    for model in models:
        subprocess.run(['ollama', 'pull', model])

if __name__ == "__main__":
    model_list_path = 'model_list'  # Replace with the actual path to your model list file
    models = read_model_list(model_list_path)
    pull_models(models)