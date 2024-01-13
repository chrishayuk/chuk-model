# training_schedule.py
from pydantic import BaseModel
from typing import List
import json

class TrainingConfig(BaseModel):
    name: str
    datafile: str
    learning_rate: float
    max_epochs: int
    patience: int
    target_validation_loss: float
    checkpoint_interval: int

class TrainingSchedule(BaseModel):
    configurations: List[TrainingConfig]

    @classmethod
    def load_from_file(cls, file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return cls(configurations=data)
