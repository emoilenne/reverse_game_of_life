from trainingModel import TrainingModel

class ModelStorage(dict):
    def __init__(self, steps, size):
        self.size = size
        self.steps = steps

    def __getitem__(self, idx):
        self.setdefault(idx, TrainingModel(steps=self.steps, size=self.size, modelHash=idx))
        return dict.__getitem__(self, idx)

    def add(self, fields, row):
        """
            Add model to models storage
        """
        model = TrainingModel(size=self.size, fields=fields, row=row)
        self[model.modelHash] = model
        return model.modelHash

    @staticmethod
    def getSteps(fields, row):
        """
            Get number of steps from input data
        """
        if 'steps' in fields:
            return int(row[fields.index('steps')])
        return None
