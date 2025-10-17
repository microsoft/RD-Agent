import torch
from .gpu_utils import setup_gpu

class GPUDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.device = setup_gpu()
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4 if torch.cuda.is_available() else 2,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def __iter__(self):
        for data, target in self.loader:
            yield data.to(self.device), target.to(self.device)

def create_gpu_optimized_loader(dataset, config=None):
    """Create data loader optimized for GPU"""
    if config is None:
        config = {}
    
    batch_size = config.get('batch_size', 32)
    if torch.cuda.is_available():
        # Increase batch size for GPU
        batch_size = batch_size * 2
        
    return GPUDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=config.get('shuffle', True)
    )