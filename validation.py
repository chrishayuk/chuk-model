import torch

class ModelValidator:
    # initialize
    def __init__(self, model, criterion, tokenizer, device):
        self.model = model
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.device = device

    def validate(self, val_dataloader):
        # stick into eval mode
        self.model.eval()

        # initialize number of batches and loss
        total_val_loss = 0
        total_val_batches = 0

        # disable gradient calculatons
        with torch.no_grad():
            # loop through the validation batches
            for input_seq, target_seq in val_dataloader:
                # get the input and target
                input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)

                # get the prediction
                output = self.model(input_seq)

                # ignore padding
                mask = (target_seq != 0).float()

                # set the loss calculation
                loss = self.criterion(output.view(-1, self.tokenizer.get_vocab_size()), target_seq.view(-1))
                masked_loss = (loss * mask.view(-1)).sum() / mask.sum()
                total_val_loss += masked_loss.item()
                total_val_batches += 1

        # calculate the average
        average_val_loss = total_val_loss / total_val_batches if total_val_batches > 0 else 0

        # return the average
        return average_val_loss
