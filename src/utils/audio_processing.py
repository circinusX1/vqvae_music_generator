import torch

# Simple system hook configuration setup placeholder
def setup_device(config_device):
    if config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
    
    
    
