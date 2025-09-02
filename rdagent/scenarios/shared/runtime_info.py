import json
import platform
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
            gpu_info["gpu_device"] = torch.cuda.get_device_name(0)
            gpu_info["total_gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            gpu_info["allocated_memory_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
            gpu_info["cached_memory_gb"] = round(torch.cuda.memory_reserved(0) / 1024**3, 2)
            gpu_info["gpu_count"] = torch.cuda.device_count()
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
                lines = result.stdout.strip().splitlines()
                gpu_info["gpus"] = []
                for line in lines:
                    name, mem_total, mem_used = [x.strip() for x in line.split(",")]
                    gpu_info["gpus"].append(
                        {
                            "name": name,
                            "memory_total_mb": int(mem_total),
                            "memory_used_mb": int(mem_used),
                        }
                    )
            else:
                gpu_info["source"] = "nvidia-smi"
                gpu_info["message"] = "No GPU detected or nvidia-smi not available"
        except FileNotFoundError:
            gpu_info["source"] = "nvidia-smi"
            gpu_info["message"] = "nvidia-smi not installed"
    return gpu_info


if __name__ == "__main__":
    info = {
        "runtime": get_runtime_info(),
        "gpu": get_gpu_info(),
    }
    print(json.dumps(info, indent=4))
