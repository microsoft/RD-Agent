import torch
import logging
import gc
import subprocess

logger = logging.getLogger(__name__)

def check_nvidia_drivers():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def is_cuda_built():
    try:
        if hasattr(torch.cuda, 'is_built'):
            return torch.cuda.is_built()
        else:
            return torch.cuda.is_available()
    except:
        return False

def setup_gpu(verbose=True):
    if verbose:
        print("Initializing GPU support...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA built with PyTorch: {is_cuda_built()}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available in PyTorch")
            print("Possible solutions:")
            print("1. Install PyTorch with CUDA support")
            print("2. Update NVIDIA drivers")
            print("3. Check CUDA toolkit installation")
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        if verbose:
            print("No GPUs detected")
        return torch.device("cpu")

    if verbose:
        print(f"Found {num_gpus} GPU(s)")

    device = torch.device("cuda:0")

    try:
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        del test_tensor
        if hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()

        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using GPU: {gpu_name}")
            print(f"GPU Memory: {memory:.1f} GB")
            try:
                if hasattr(torch.version, 'cuda'):
                    print(f"CUDA version: {torch.version.cuda}")
            except:
                print("CUDA version: Unknown")

        if hasattr(torch.backends, 'cudnn'):
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'deterministic'):
                torch.backends.cudnn.deterministic = False

        return device

    except Exception as e:
        if verbose:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU")
        return torch.device("cpu")

def force_cuda_initialization():
    if torch.cuda.is_available():
        try:
            x = torch.cuda.FloatTensor(1)
            del x
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
            return True
        except Exception as e:
            print(f"CUDA forced initialization failed: {e}")
            return False
    return False

def get_gpu_info():
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_built": is_cuda_built(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": []
    }

    try:
        if hasattr(torch.version, 'cuda'):
            info["cuda_version"] = torch.version.cuda
        else:
            info["cuda_version"] = "Unknown"
    except:
        info["cuda_version"] = "Unknown"

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                gpu_info = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
                }
                try:
                    gpu_info["memory_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
                    gpu_info["memory_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9
                except:
                    gpu_info["memory_allocated_gb"] = 0
                    gpu_info["memory_reserved_gb"] = 0
                info["gpus"].append(gpu_info)
            except Exception as e:
                print(f"Could not get info for GPU {i}: {e}")

    return info

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.1f}GB")
            except Exception as e:
                print(f"Could not get memory info for GPU {i}: {e}")

def clear_gpu_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Could not clear GPU cache: {e}")

def optimize_model_for_gpu(model):
    if torch.cuda.is_available():
        try:
            model = model.cuda()
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                    print("Model compilation enabled")
                except Exception as e:
                    print(f"Model compilation failed: {e}")
        except Exception as e:
            print(f"Failed to move model to GPU: {e}")
    return model

def check_pytorch_installation():
    print("PyTorch Installation Check")
    print("=" * 40)
    print(f"Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Built with CUDA: {is_cuda_built()}")

    if not torch.cuda.is_available():
        print("\nRECOMMENDATION:")
        print("To enable GPU support, install PyTorch with CUDA:")
        print("For CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("For CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    return torch.cuda.is_available()
