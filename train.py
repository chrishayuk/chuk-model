from os import device_encoding
from data_loader import load_data_from_jsonl
from tokenized_pair_dataset import TokenizedPairDataset
from tokenizer import tokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_scheduler import TrainingScheduler
from model.model_loader import load_model_from_file
from model.model_state_manager import ModelStateManager
from model.model_checkpoint_manager import ModelCheckpointManager
from utils import get_best_device
from validation import ModelValidator
from early_stopping import EarlyStopping
from batch_size_optimizer import find_optimal_batch_size

# loads a checkpoint
def load_checkpoint(model, optimizer, checkpoint_manager, schedule_name, schedule_epoch, device):
    # set the start epoch as zero
    start_epoch = 0

    # Set device before loading the checkpoint
    model = model.to(device)

    # Get the latest checkpoint
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint(schedule_name, schedule_epoch)

    # is latest checkpoint
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")

        try:
            # Load the checkpoint
            model, optimizer_state_dict, last_completed_epoch, _ = checkpoint_manager.load_checkpoint(model, latest_checkpoint, device)
            
            # Load the optimizer state
            optimizer.load_state_dict(optimizer_state_dict)

            # Update the start epoch for resuming training
            start_epoch = last_completed_epoch + 1
            print(f"Resuming training from epoch {start_epoch}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # return model etc
    return model, optimizer, start_epoch


# prepare the data for training
def prepare_data(config, max_seq_length, batch_size):
    # load the data from jsonl
    input_strings, target_strings = load_data_from_jsonl(config.datafile)

    # setup the splits
    split_ratio = 0.8
    split_index = int(len(input_strings) * split_ratio)

    # get the training data with the split
    train_input_strings = input_strings[:split_index]
    train_target_strings = target_strings[:split_index]

    # get the validation input with the split
    val_input_strings = input_strings[split_index:]
    val_target_strings = target_strings[split_index:]

    # load the training and validation dataset
    train_dataset = TokenizedPairDataset(train_input_strings, train_target_strings, max_seq_length)
    val_dataset = TokenizedPairDataset(val_input_strings, val_target_strings, max_seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # return the data
    return train_dataloader, val_dataloader


# Load the model using the model loader
model = load_model_from_file("./model_config.json").to(get_best_device())
criterion = nn.CrossEntropyLoss()

# Print model configuration details (optional)
print(f"Model Configuration: {model}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# get the device
device = get_best_device()
print(f"Device: {device}")

# #find the optimal batch size
# max_seq_length = 200
# batch_test_input_strings, batch_test_target_strings = load_data_from_jsonl("./data/counting_broad.jsonl")
# batch_test_dataset = TokenizedPairDataset(batch_test_input_strings, batch_test_target_strings, max_seq_length)

# print(f"Attempting to find optimal batch size")
# optimal_batch_size = find_optimal_batch_size(model, batch_test_dataset, device=device)
# print(f"Optimal Batch Size: {optimal_batch_size}")

# Set up the ModelStateManager and ModelCheckpointManager
state_manager = ModelStateManager(model.__class__, "./saved_models", device)
checkpoint_manager = ModelCheckpointManager(model.__class__, "./checkpoints")

# Initialize the training scheduler with the configuration file
scheduler = TrainingScheduler("training_scheduler_config.json")

# initalize the validator
validator = ModelValidator(model, criterion, tokenizer, device)

# schedule epochs
for schedule_epoch in range(0, 1):
    # Iterate over each configuration in the schedule
    for config in scheduler.get_schedule():
        # Training setup
        print(f"Training schedule: {config.name}")

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Load checkpoint
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_manager, config.name, schedule_epoch, device)

        # Load data from the specified file in the configuration
        input_strings, target_strings = load_data_from_jsonl(config.datafile)

        # Dataset and DataLoader setup
        batch_size = 64
        max_seq_length = 200 # todo: need to get this from the model

        # Get max_epochs and target_validation_loss from the config
        max_epochs = config.max_epochs
        target_validation_loss = config.target_validation_loss

        # Load and prepare data
        train_dataloader, val_dataloader = prepare_data(config, max_seq_length, batch_size)

        # get the checkpoint interval
        checkpoint_interval = getattr(config, 'checkpoint_interval', config.max_epochs)

        # set early stopping
        early_stopper = EarlyStopping(patience=config.patience)

        # Training loop
        for epoch in range(start_epoch, max_epochs):
            model.train()
            total_loss = 0
            total_items_processed = 0
            for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
                optimizer.zero_grad()
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                #print(input_seq)
                output = model(input_seq)
                assert output.shape[:2] == target_seq.shape, f"Output shape {output.shape} does not match target shape {target_seq.shape}"
                mask = (target_seq != 0).float()
                loss = criterion(output.view(-1, tokenizer.get_vocab_size()), target_seq.view(-1))
                masked_loss = (loss * mask.view(-1)).sum() / mask.sum()
                masked_loss.backward()
                optimizer.step()
                total_loss += masked_loss.item()
                total_items_processed += input_seq.size(0) 

                # Check and print every 100 items
                if total_items_processed % 320 == 0:
                    print(f"Processed {total_items_processed} items, Current Batch Loss: {masked_loss.item()}")

            average_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} - Average Training Loss: {average_loss}")

            # Validation
            average_val_loss = validator.validate(val_dataloader)
            print(f"Epoch {epoch+1} - Average Validation Loss: {average_val_loss}")

            # Check if target validation loss reached
            if average_val_loss <= target_validation_loss:
                print(f"Target validation loss reached at epoch {epoch + 1}. Saving checkpoint and stopping training.")
                checkpoint_manager.save_checkpoint(model, optimizer, epoch + 1, average_val_loss, config.name, schedule_epoch)
                break

            # check for early stopping
            early_stopper(average_val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                checkpoint_manager.save_checkpoint(model, optimizer, epoch + 1, average_val_loss, config.name, schedule_epoch)
                break

            # Regular checkpointing logic
            if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == max_epochs:
                print(f"Saving checkpoint for {config.name} at epoch {epoch + 1}")
                checkpoint_manager.save_checkpoint(model, optimizer, epoch + 1, average_loss, config.name, schedule_epoch)



# Final model save (optional)
print(f"saving final model")
state_manager.save_state(model, "trained_model_final.pth")
