import numpy as np
import window

class ModelStorage(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

class TrainingModel:
    def __init__(self, window):
        """
            Initiate training model, which stores predictions for start grid and number of
            occurances of this model in the testing.
        """
        self.steps = steps
        self.occurances = 0
        self.data = window.data
        self.height = window.height
        self.width = window.width

    def addOccurance(self, window):
        """
            Add start grid values of this model to predictions.
        """
        self.occurances += 1
        self.data += window.data

    def predict(self):
        """
            Return predictions array, that was formed on the training data.
        """
        return self.data

class TrainingAgent:
    def __init__(self, height=20, width=20, windowSize=4):
        """
            Initiate training agent by grid dimentions, size of the window that
            will scan training data and create storage for training models.
        """
        self.height = height
        self.width = width
        self.windowSize = windowSize
        self.models = ModelStorage()

    def trainWindow(startPoint, grid):
        """
            Determine window's model and add predictions to training models.
        """
        trainingWindow = window.Window(grid, startPoint, self.windowSize)
        

    def train(self, rows):
        """
            Train agent to predict start grid based on training data.
        """
