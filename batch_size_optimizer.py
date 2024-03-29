import torch
from torch.utils.data import DataLoader
from utils import get_best_device

def test_batch_size(model, data_loader, device):
    try:
        for input_seq, target_seq in data_loader:
            # get the input and the target
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # performs a forward pass
            model(input_seq)
        return True
    except RuntimeError as e:
        # check for out of memory
        if 'out of memory' in str(e):
            return False
        raise e

def find_optimal_batch_size(model, dataset, start_batch_size=32, max_batch_size=2048, device=None):
    # check if the device has been specified, if not get the best device
    if device is None:
        device = get_best_device()

    # set the device for the model
    model.to(device)

    # set the optimal batch size
    optimal_batch_size = start_batch_size
    current_batch_size = start_batch_size

    # print the current batch size
    print(f"Device: {device}")
    print(f"Starting Batch Size: {start_batch_size}")
    
    # loop from current batch size, to max batch size
    while current_batch_size <= max_batch_size:
        # instantiate the data loader
        data_loader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=True)

        # loading new batch size
        print(f"Attempting Batch Size: {current_batch_size}")

        # test the barch size
        if test_batch_size(model, data_loader, device):
            # success, print the batch size
            print(f"Succesful Batch Size: {current_batch_size}")

            # increment the batch size
            optimal_batch_size = current_batch_size
            current_batch_size *= 2
        else:
            # success, print the batch size
            print(f"Failed Batch Size: {current_batch_size}")

            # returns the last successful batch size
            return optimal_batch_size