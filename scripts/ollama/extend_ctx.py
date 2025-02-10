import subprocess
import json
model_list = []
# Read model_list file
with open('model_list', 'r') as file:
    # One model_name per line, strip newlines
    model_list = [ model_name.strip() for model_name in file.readlines()]
# model_list = ['deepseek-r1:32b']
print (model_list)

# Iterate through model_list
for model_name in model_list:
    # Read modelfile
    with open('ModelFile', 'w') as file:
        file.write(f"FROM {model_name}\n")
        file.write("PARAMETER num_ctx 16384\n")
    
    
    # Execute ollama create command
    new_model_name = f"{model_name}-16k"
    subprocess.run(['ollama', 'create', new_model_name, '-f', 'ModelFile'])