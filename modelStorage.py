from trainingModel import TrainingModel

class ModelStorage(dict):
    def __init__(self, steps):
        self.steps = steps

    def __getitem__(self, idx):
        self.setdefault(idx, TrainingModel(steps=self.steps, neighbors=idx))
        return dict.__getitem__(self, idx)

    def add(self, fields, row):
        """
            Add model to models storage
        """
        # Create model from row
        model = TrainingModel(fields=fields, row=row)

        # Save model
        self[model.neighbors] = model

        return model.neighbors

    @staticmethod
    def getSteps(fields, row):
        """
            Get number of steps from input data
        """
        if 'steps' in fields:
            return int(row[fields.index('steps')])
        return None
