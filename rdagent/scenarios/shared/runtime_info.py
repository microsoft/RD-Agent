import json
import platform
import re
import subprocess
import sys
from importlib.metadata import distributions


def get_runtime_info():
    return {
        "python_version": sys.version,
        "os": platform.system(),
        "os_release": platform.release(),
    }


def get_gpu_info():
    gpu_info = {}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["source"] = "pytorch"
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["gpu_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                gpu_name_list = []
                gpu_total_mem_list = []
                gpu_allocated_mem_list = []

                for i in range(torch.cuda.device_count()):
                    gpu_name_list.append(torch.cuda.get_device_name(i))
                    gpu_total_mem_list.append(torch.cuda.get_device_properties(i).total_memory)
                    gpu_allocated_mem_list.append(torch.cuda.memory_allocated(i))

                gpu_info["gpus"] = []
                for i in range(torch.cuda.device_count()):
                    gpu_info["gpus"].append(
                        {
                            "index": i,
                            "name": gpu_name_list[i],
                            "memory_total_gb": round(gpu_total_mem_list[i] / 1024**3, 2),
                            "memory_used_gb": round(gpu_allocated_mem_list[i] / 1024**3, 2),
                        }
                    )
                gpu_info["summary"] = {
                    "gpu_count": torch.cuda.device_count(),
                    "total_memory_gb": round(sum(gpu_total_mem_list) / 1024**3, 2),
                    "total_used_memory_gb": round(sum(gpu_allocated_mem_list) / 1024**3, 2),
                }
            else:
                gpu_info["message"] = "No CUDA GPU detected (PyTorch)"
        else:
            gpu_info["source"] = "pytorch"
            gpu_info["message"] = "No CUDA GPU detected"
    except ImportError:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                gpu_info["source"] = "nvidia-smi"
                gpu_info["cuda_version"] = None
                version_result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                )
                if version_result.returncode == 0:
                    match = re.search(r"CUDA Version:\s*([0-9.]+)", version_result.stdout)
                    if match:
                        gpu_info["cuda_version"] = match.group(1)
                lines = result.stdout.strip().splitlines()
                gpu_info["gpus"] = []
                total_mem_list = []
                used_mem_list = []
                for index, line in enumerate(lines):
                    name, mem_total, mem_used = [x.strip() for x in line.split(",")]
                    total_mem_list.append(int(mem_total))
                    used_mem_list.append(int(mem_used))
                    gpu_info["gpus"].append(
                        {
                            "index": index,
                            "name": name,
                            "memory_total_gb": round(int(mem_total) / 1024, 2),
                            "memory_used_gb": round(int(mem_used) / 1024, 2),
                        }
                    )
                gpu_info["gpu_count"] = len(gpu_info["gpus"])
                gpu_info["summary"] = {
                    "gpu_count": len(gpu_info["gpus"]),
                    "total_memory_gb": round(sum(total_mem_list) / 1024, 2),
                    "total_used_memory_gb": round(sum(used_mem_list) / 1024, 2),
                }
            else:
                gpu_info["source"] = "nvidia-smi"
                gpu_info["cuda_version"] = None
                gpu_info["message"] = "No GPU detected or nvidia-smi not available"
        except FileNotFoundError:
            gpu_info["source"] = "nvidia-smi"
            gpu_info["cuda_version"] = None
            gpu_info["message"] = "nvidia-smi not installed"
    return gpu_info


if __name__ == "__main__":
    info = {
        "runtime": get_runtime_info(),
        "gpu": get_gpu_info(),
    }
    print(json.dumps(info, indent=4))
