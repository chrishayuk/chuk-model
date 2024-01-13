from training_schedule import TrainingSchedule

class TrainingScheduler:
    def __init__(self, config_file):
        self.schedule = TrainingSchedule.load_from_file(config_file)

    def get_schedule(self):
        for config in self.schedule.configurations:
            yield config