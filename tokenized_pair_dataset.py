import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizer.tokenizer import encode

class TokenizedPairDataset(Dataset):
    def __init__(self, input_strings, target_strings, max_seq_length):
        # Store the encoded input and target strings
        self.input_data = [encode(s) for s in input_strings]
        self.target_data = [encode(s) for s in target_strings]

        # Compute the maximum sequence length
        self.max_seq_length = max_seq_length

    def __len__(self):
        # returns the length of the dataset
        return len(self.input_data)
    

    def __getitem__(self, idx):
        # Retrieve the encoded input and target sequences
        input_seq = self.input_data[idx]
        target_seq = self.target_data[idx]

        # Pad both input and target sequences to the same maximum length
        padded_input_seq = input_seq + [0] * (self.max_seq_length - len(input_seq))
        padded_target_seq = target_seq + [0] * (self.max_seq_length - len(target_seq))

        # Create tensors from the padded sequences
        input_tensor = torch.tensor(padded_input_seq)
        target_tensor = torch.tensor(padded_target_seq)

        # Return the input and target tensors
        return input_tensor, target_tensor

if __name__ == "__main__":
    # some example strings to reverse, and their targets
    input_strings = ["reverse:short", "reverse:longer", "reverse:muchmuchlonger"]
    target_strings = ["trohs", "regnol", "regnolhcumhcum"]

    # load the examples into the dataset
    dataset = TokenizedPairDataset(input_strings, target_strings, 20)

    # load the dataset in batches
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # output
    for input_seq, target_seq in dataloader:
        print("Input Sequence:", input_seq)
        print("Target Sequence:", target_seq)
        print("Max Sequence Length:", dataset.max_seq_length)