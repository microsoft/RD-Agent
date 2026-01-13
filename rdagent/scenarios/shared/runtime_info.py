import platform
import subprocess
import sys
from importlib.metadata import distributions


def print_runtime_info():
    print("=== Python Runtime Info ===")
    print(f"Python {sys.version} on {platform.system()} {platform.release()}")


def get_gpu_info():
    try:
        # Option 1: Use PyTorch
        import torch

        if torch.cuda.is_available():
            print("\n=== GPU Info (via PyTorch) ===")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                gpu_name_list = []
                gpu_total_mem_list = []
                gpu_allocated_mem_list = []
                gpu_cached_mem_list = []

                for i in range(torch.cuda.device_count()):
                    gpu_name_list.append(torch.cuda.get_device_name(i))
                    gpu_total_mem_list.append(torch.cuda.get_device_properties(i).total_memory)
                    gpu_allocated_mem_list.append(torch.cuda.memory_allocated(i))
                    gpu_cached_mem_list.append(torch.cuda.memory_reserved(i))

                for i in range(torch.cuda.device_count()):
                    print(f"  - GPU {i}: {gpu_name_list[i]}")
                    print(f"    Total Memory: {gpu_total_mem_list[i] / 1024**3:.2f} GB")
                    print(f"    Allocated Memory: {gpu_allocated_mem_list[i] / 1024**3:.2f} GB")
                    print(f"    Cached Memory: {gpu_cached_mem_list[i] / 1024**3:.2f} GB")
                print("  - All GPUs Summary:")
                print(f"    Total Memory: {sum(gpu_total_mem_list) / 1024**3:.2f} GB")
                print(f"    Total Allocated Memory: {sum(gpu_allocated_mem_list) / 1024**3:.2f} GB")
                print(f"    Total Cached Memory: {sum(gpu_cached_mem_list) / 1024**3:.2f} GB")
            else:
                print("No CUDA GPU detected (PyTorch)!")
        else:
            print("\nNo CUDA GPU detected (PyTorch).")

    except ImportError:
        # Option 2: Use nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("\n=== GPU Info (via nvidia-smi) ===")
                print(result.stdout.strip())
            else:
                print("\nNo GPU detected (nvidia-smi not available).")
        except FileNotFoundError:
            print("\nNo GPU detected (nvidia-smi not installed).")


if __name__ == "__main__":
    print_runtime_info()
    get_gpu_info()
