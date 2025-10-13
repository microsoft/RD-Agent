import torch
import torch.nn as nn
import sys
import os
from rdagent.app.general_model.general_model import GPUEnhancedLSTM

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from rdagent.app.utils.gpu_utils import setup_gpu, print_gpu_memory, get_gpu_info, force_cuda_initialization, check_pytorch_installation

def comprehensive_gpu_test():
    print(" Comprehensive GPU Support Test")
    print("=" * 60)
    
    gpu_available = check_pytorch_installation()
    
    print("\n" + "=" * 60)
    
    gpu_info = get_gpu_info()
    print(f" PyTorch Version: {gpu_info['pytorch_version']}")
    print(f" CUDA Built: {gpu_info['cuda_built']}")
    print(f" CUDA Available: {gpu_info['cuda_available']}")
    print(f"GPU Count: {gpu_info['gpu_count']}")
    
    if gpu_info['cuda_available']:
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f" GPU {i}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_total_gb']:.1f} GB")
    
    print("\n" + "=" * 60)
    
    # Test device setup
    print("\n1. Testing GPU Setup...")
    device = setup_gpu(verbose=True)
    print(f" Final device: {device}")
    
    # Force CUDA initialization
    print("\n2. Testing CUDA Initialization...")
    cuda_working = force_cuda_initialization()
    print(f"CUDA working: {cuda_working}")
    
    # Test model creation and movement
    print("\n3. Testing Model Creation...")
    try:        
        model = GPUEnhancedLSTM(10, 50, 2, 1)
        print(f" Model created on: {next(model.parameters()).device}")
        
        # Test if we can move to GPU
        if torch.cuda.is_available():
            model = model.to(device)
            print(f"Model moved to: {next(model.parameters()).device}")
        else:
            print(" Skipping model movement (no GPU available)")
            
    except Exception as e:
        print(f" Model test failed: {e}")
        # Create a simple fallback model for testing
        try:
            class SimpleLSTM(nn.Module):
                def __init__(self):
                    super(SimpleLSTM, self).__init__()
                    self.lstm = nn.LSTM(10, 50, 2, batch_first=True)
                    self.fc = nn.Linear(50, 1)
                
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])
            
            model = SimpleLSTM()
            print(f"Fallback model created on: {next(model.parameters()).device}")
            if torch.cuda.is_available():
                model = model.to(device)
                print(f"Fallback model moved to: {next(model.parameters()).device}")
        except Exception as e2:
            print(f"Fallback model also failed: {e2}")
    
    # Test data movement
    print("\n4. Testing Data Transfer...")
    try:
        test_tensor = torch.randn(32, 10, 10)
        print(f"Tensor created on: {test_tensor.device}")
        
        if torch.cuda.is_available():
            test_tensor = test_tensor.to(device)
            print(f"Tensor moved to: {test_tensor.device}")
    except Exception as e:
        print(f" Data transfer test failed: {e}")
    
    # Test memory operations
    print("\n5. Testing GPU Memory...")
    print_gpu_memory()
    
    # Performance test (only if GPU is available)
    print("\n6. Basic Performance Test...")
    if torch.cuda.is_available():
        try:
            # Simple matrix multiplication test
            size = 1000
            a = torch.randn(size, size).to(device)
            b = torch.randn(size, size).to(device)
            
            import time
            
            # Warm up
            for _ in range(3):
                _ = torch.matmul(a, b)
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
            
            # GPU timing
            start_time = time.time()
            for _ in range(10):
                c = torch.matmul(a, b)
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / 10
            
            # CPU timing
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            start_time = time.time()
            for _ in range(10):
                c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = (time.time() - start_time) / 10
            
            print(f"  GPU time: {gpu_time:.4f}s")
            print(f"  CPU time: {cpu_time:.4f}s")
            if gpu_time > 0:
                print(f" Speedup: {cpu_time/gpu_time:.2f}x")
            
        except Exception as e:
            print(f" Performance test failed: {e}")
    else:
        print("Skipping performance test (no GPU available)")
    
    print("\n" + "=" * 60)
    print(" GPU Support Test Completed!")
    
    # Final status
    if torch.cuda.is_available():
        print("GPU support is WORKING!")
    else:
        print(" GPU support is NOT available")
        print("\n To enable GPU support:")
        print("1. Check if you have an NVIDIA GPU")
        print("2. Install NVIDIA drivers")
        print("3. Install PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    comprehensive_gpu_test()