from trainingModel import TrainingModel

class ModelStorage(dict):
    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        self.setdefault(idx, TrainingModel(self.size))
        return dict.__getitem__(self, idx)
