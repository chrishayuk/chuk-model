import os
import torch
from utils import get_best_device
from model.model_config_manager import ModelConfigManager
from model.model_checkpoint_manager import ModelCheckpointManager
from .transformer_model import TransformerModel

# loads a model from a file
def load_model_from_file(full_config_path):
    # Split the full path into directory and filename
    config_dir = os.path.dirname(full_config_path)
    config_filename = os.path.basename(full_config_path)

    # Initialize ModelConfigManager with the directory path
    model_config_manager = ModelConfigManager(config_dir)

    # Load the configuration using the filename
    model_config = model_config_manager.load_config(config_filename)

    # Load and return the model
    return load_model(model_config)
# loads the model and returns a transformer
def load_model(model_config, device=None):
    # get the best device if not set
    if device is None:
        device = get_best_device()

    # load the transfomer
    model = TransformerModel(model_config.vocab_size, model_config.d_model, model_config.nhead, model_config.num_decoder_layers, model_config.dim_feedforward, model_config.max_seq_length)

    # set the device
    model.to(device)

    # return the model
    return model

# loads the model from the checkpoint
def load_model_from_checkpoint(checkpoint_path, device=None):
    # checkpoint manager
    checkpoint_manager = ModelCheckpointManager(TransformerModel, checkpoint_path)

    # load the checkpoint
    model_config, optimizer_state_dict, epoch, loss = checkpoint_manager.load_checkpoint(checkpoint_path)

    # load the model
    model = load_model(model_config, device)

    # load the state
    model.load_state_dict(optimizer_state_dict)

    # return model etc
    return model, optimizer_state_dict, epoch, loss