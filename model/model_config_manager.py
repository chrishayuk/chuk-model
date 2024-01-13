import json
import os
from .model_config import ModelConfig

class ModelConfigManager:
    # setup the location of the model config
    def __init__(self, config_path):
        # set the model config path
        self.config_path = config_path

        # create the directory if it doesn't exit
        os.makedirs(self.config_path, exist_ok=True)

    # save the model config as json
    def save_config(self, config, config_name="model_config.json"):
        # conver the config to a dictionary
        config_data = config if isinstance(config, dict) else config.__dict__

        # save the config
        with open(os.path.join(self.config_path, config_name), 'w') as f:
            json.dump(config_data, f)

    # load the model config from json
    def load_config(self, config_name="model_config.json"):
        # get the file path by joining the config name, with the path
        file_path = os.path.join(self.config_path, config_name)

        # open the file
        with open(file_path, 'r') as file:
            # load the config
            config_data = json.load(file)

        # return the model config
        return ModelConfig(**config_data)
