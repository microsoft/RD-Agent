import platform
import subprocess
import sys
from importlib.metadata import distributions


def print_runtime_info():
    print(f"Python {sys.version} on {platform.system()} {platform.release()}")


def get_installed_packages():
    return {dist.metadata["Name"].lower(): dist.version for dist in distributions()}


def print_filtered_packages(installed_packages, filtered_packages):
    for package_name in filtered_packages:
        version = installed_packages.get(package_name.lower())
        if version:
            print(f"{package_name}=={version}")


def get_gpu_info():
    try:
        # Option 1: Use PyTorch
        import torch

        if torch.cuda.is_available():
            print("\n=== GPU Info (via PyTorch) ===")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"Cached Memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
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
    filtered_packages = [
        "transformers",
        "accelerate",
        "torch",
        "tensorflow",
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "lightgbm",
        "vtk",
        "opencv-python",
        "keras",
        "matplotlib",
        "pydicom",
    ]
    installed_packages = get_installed_packages()
    print_filtered_packages(installed_packages, filtered_packages)
    get_gpu_info()
