from utils import get_best_device
import torch
import os

class ModelStateManager:
    # set the model class, and path
    def __init__(self, model_class, state_path, device=None):
        self.model_class = model_class
        self.state_path = state_path
        
        # get the best device if not set
        if device is None:
            device = get_best_device()

        # if the model directory doesn't exist, create it
        os.makedirs(self.state_path, exist_ok=True)

    # saves the weights and biases
    def save_state(self, model, state_name):
        # save the weights and biases
        torch.save(model.state_dict(), os.path.join(self.state_path, state_name))

    # load the weights and biases
    def load_state(self, state_name):
        # instantiate the model class
        model = self.model_class()

        # load the weights and biases
        model.load_state_dict(torch.load(os.path.join(self.state_path, state_name)))

        # return the model
        return model
