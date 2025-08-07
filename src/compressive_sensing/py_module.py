import torch
from ._compressive_sensing import hello

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)

    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

def hello_world():
    """A simple Python wrapper around the C++ hello function
    that automatically uses the best available device
    """
    device = get_device()
    print(f"Python wrapper calling C++ function with device: {device}")
    hello(device)
    return device
