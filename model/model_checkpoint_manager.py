import torch
import os
import re

class ModelCheckpointManager:
    def __init__(self, model_class, checkpoint_path):
        self.model_class = model_class
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def get_latest_checkpoint(self, schedule_name, schedule_epoch):
        # Regex pattern to match filename with schedule_name and schedule_epoch
        pattern = re.compile(rf"{schedule_name}_epoch_\d+_schedepoch_{schedule_epoch}\.pth")

        # Filter checkpoint files based on the pattern
        checkpoint_files = [f for f in os.listdir(self.checkpoint_path) if pattern.match(f)]

        if not checkpoint_files:
            return None

        return max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(self.checkpoint_path, f)))
    
    def save_checkpoint(self, model, optimizer, epoch, loss, schedule_name, schedule_epoch):
        checkpoint_name = f"{schedule_name}_epoch_{epoch}_schedepoch_{schedule_epoch}.pth"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_path, checkpoint_name))
    
    def load_checkpoint(self, model, checkpoint_name, device):
        checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', None)
        return model, optimizer_state_dict, epoch, loss
