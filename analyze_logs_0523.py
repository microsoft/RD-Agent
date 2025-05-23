import pickle
import tarfile
import re

competition_names = [
    "aerial-cactus-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "hotel-id-2021-fgvc8",
    "mlsp-2013-birds",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "text-normalization-challenge-english-language"
]

root_path = "/mnt/c/Users/yugzhan/OneDrive - Microsoft/rdagent-logs/yuge20250522"

for competition_name in competition_names:
    tar_path = f"{root_path}/{competition_name}.log.tar"

    log_pattern = f"log/{competition_name}/" + r"Loop_(\d+)/feedback/llm_messages/\d+/common_logs.log"

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if match := re.match(log_pattern, member.name):
                print(member.name)
                loop_id = int(match.group(1))
                file = tar.extractfile(member)
                if file:
                    file_content = file.read()
                    with open(f"logs_0523/feedback_{competition_name}_loop{loop_id:02d}.log", "wb") as f:
                        f.write(file_content)
