import torch

# get the best device
def get_best_device():
    # set the device
    device = torch.device("cpu")

    # check for cuda
    if torch.cuda.is_available():
        # cuda (gpu)
        device = torch.device("cuda")
    # Check for MPS (Apple Silicon) availability
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # return the device
    return device