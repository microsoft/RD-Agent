import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to load and process each pickle file
def process_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        # Assuming data is a DataFrame or similar
        print(f"Data from {file_path} processed successfully.")
        return data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def analysis(folder_path):
    success_count = 0
    fail_count = 0

    # Logging the errors
    error_log = open("error_log.log", "w")

    # List to store data for visualization
    data_frames = []

    # Processing each file in the directory
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        data = process_pickle_file(file_path)
        if data is not None:
            data_frames.append(data)

    for df in data_frames:
        if 'Execution succeeded' in df[0]:
            success_count += 1
        else:
            fail_count += 1
            error_log.write(f"{file_path}: \n{df[0]}\n")

    # Writing summary
    print(f"Number of successful files: {success_count}")
    print(f"Number of failed files: {fail_count}")

    # Closing the error log file
    error_log.close()

def view_pickle_file(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        print(f'the path of this file is: {file_path}\n')
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            for i in range(len(data)):
                print(data[i])


if __name__ == '__main__':
    folder_path = '/data/userdata/v-taozhiwang/RD-Agent/git_ignore_folder/factor_implementation_execution_cache'
    
    analysis(folder_path)